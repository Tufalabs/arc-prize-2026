"""Tests for taaf.diagnostics — the scorers, the answer sheet that replaced
the movies, the single-step lane curve, the summary's honesty about
unscored runs, and games still in play read at the answer they hold."""

from __future__ import annotations

import base64
import copy
import dataclasses
import io
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.image
import numpy as np
import pytest

import taaf.benchmark
import taaf.diagnostics
import taaf.game_examples
import taaf.grid
import taaf.solver_examples
import taaf.support
from taaf.game import ArcGame, GameSpec, Pair, Submission
from taaf.grid import Grid


def _multi_spec(game_id: str = "m") -> GameSpec:
    """A task with three pairs to answer, so partial credit is visible."""
    pairs = [Pair(question=Grid(np.array([[1]], np.int8)), answer=Grid(np.array([[2]], np.int8)), given=True)]
    pairs += [
        Pair(question=Grid(np.array([[3]], np.int8)), answer=Grid(np.array([[4]], np.int8)), given=False)
        for _ in range(3)
    ]
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def _played(spec: GameSpec, n_right: int, tokens: int = 10, withhold: bool = False):
    """Play ``spec`` answering ``n_right`` of its pairs correctly."""
    from taaf.game import RunSession

    game = ArcGame(spec_data=spec)
    game.start_game(RunSession(withhold_answers=withhold))
    right, wrong = Grid(np.array([[4]], np.int8)), Grid(np.array([[0]], np.int8))
    attempts = tuple((right, right) if i < n_right else (wrong, wrong) for i in range(len(spec.to_answer)))
    game.execute_action(Submission(attempts=attempts), generated_tokens=tokens)
    game.finish_game()
    assert game.game_run is not None
    return game.game_run


def _played_sequence(spec: GameSpec, rights: tuple[int, ...], tokens: int = 100):
    """Play ``spec`` submitting one answer per entry in ``rights``, each
    getting that many pairs right and costing ``tokens``."""
    from taaf.game import RunSession

    game = ArcGame(spec_data=spec)
    game.start_game(RunSession())
    right, wrong = Grid(np.array([[4]], np.int8)), Grid(np.array([[0]], np.int8))
    for n_right in rights:
        attempts = tuple((right, right) if i < n_right else (wrong, wrong) for i in range(len(spec.to_answer)))
        game.execute_action(Submission(attempts=attempts), generated_tokens=tokens)
    game.finish_game()
    assert game.game_run is not None
    return game.game_run


def test_arc_scorer_is_the_fraction_and_solved_is_all_or_nothing():
    """The two scorers differ on exactly the multi-test tasks, which is 37.5%
    of the public evaluation set — so reporting one is a choice."""
    partial = _played(_multi_spec(), n_right=2)
    full = _played(_multi_spec(), n_right=3)
    assert taaf.diagnostics.ARC_SCORER.of_run(partial) == pytest.approx(200 / 3)
    assert taaf.diagnostics.SOLVED_SCORER.of_run(partial) == 0.0
    assert taaf.diagnostics.ARC_SCORER.of_run(full) == 100.0
    assert taaf.diagnostics.SOLVED_SCORER.of_run(full) == 1.0


def test_scorers_report_zero_for_an_unscored_run_but_final_score_stays_none():
    """Plotting needs a number; the distinction between unknown and zero is
    kept on the score the run reports, which is what the summary reads."""
    run = _played(_multi_spec(), n_right=3, withhold=True)
    assert run.final_score is None and run.current_score() is None
    assert taaf.diagnostics.ARC_SCORER.of_run(run) == 0.0


def test_a_lane_that_answers_once_steps_once():
    """The floor case: one answer, one step, at what the lane spent."""
    run = _played(_multi_spec(), n_right=3, tokens=250)
    events = taaf.diagnostics._per_game_partial_curve(run, taaf.diagnostics.TOKEN_AXIS)
    assert events == [(0.0, 0.0), (250.0, 100.0)]


def test_a_lane_steps_at_every_answer_and_can_step_back_down():
    """Replacing a right answer with a wrong one is a real thing a solver can
    do, and the lane has to say so — scores along a lane are not monotonic."""
    run = _played_sequence(_multi_spec(), rights=(3, 0, 3), tokens=100)
    events = taaf.diagnostics._per_game_partial_curve(run, taaf.diagnostics.TOKEN_AXIS)
    assert events == [(0.0, 0.0), (100.0, 100.0), (200.0, 0.0), (300.0, 100.0)]


def test_lane_score_at_budget_reads_the_answer_that_stood_there():
    """A budget cuts a lane where it cuts it: what counts is the answer
    standing at that spend, not the one the run eventually settled on."""
    run = _played_sequence(_multi_spec(), rights=(3, 0, 3), tokens=100)
    at = taaf.diagnostics.lane_score_at_budget
    assert at(run, 150.0, taaf.diagnostics.ARC_SCORER) == pytest.approx(100.0)
    assert at(run, 250.0, taaf.diagnostics.ARC_SCORER) == pytest.approx(0.0)
    assert at(run, 300.0, taaf.diagnostics.ARC_SCORER) == pytest.approx(100.0)


def test_pooling_holds_each_lane_at_its_last_answer_and_stops_charging_it():
    """The combiner by hand, on a lane that regresses and a lane that ends
    early: mean cost clamps each lane at its own total, mean score holds each
    lane's last value — including when that value went down."""
    regressing = [(0.0, 0.0), (10.0, 100.0), (20.0, 0.0)]
    early = [(0.0, 0.0), (5.0, 50.0)]
    xs, ys = taaf.diagnostics._pool_lane_events([regressing, early])
    assert [round(x, 6) for x in xs] == [0.0, 5.0, 7.5, 12.5]
    assert [round(y, 6) for y in ys] == [0.0, 25.0, 75.0, 25.0]


def test_pooled_curve_rises_as_lanes_finish():
    """Three lanes ending at different costs give three knots — the shape the
    budget reading depends on."""
    runs = [_played(_multi_spec(f"g{i}"), n_right=3, tokens=100 * (i + 1)) for i in range(3)]
    xs, ys = taaf.diagnostics._pass_curve(runs, taaf.diagnostics.TOKEN_AXIS)
    assert list(xs) == sorted(xs)
    assert list(ys) == sorted(ys)
    assert ys[0] == 0.0
    assert ys[-1] == pytest.approx(100.0)


def test_weights_shift_the_pooled_score_toward_multi_test_tasks():
    """Equal weights give the per-task mean; test-output weights give
    Kaggle's per-output mean. They differ, which is the whole point."""
    one = _played(taaf.game_examples.flip_task("one", n_to_answer=1), n_right=0, tokens=10)
    three = _played(_multi_spec("three"), n_right=3, tokens=10)
    _, equal = taaf.diagnostics._pool_lane_events(
        [taaf.diagnostics._per_game_partial_curve(r, taaf.diagnostics.TOKEN_AXIS) for r in (one, three)]
    )
    _, weighted = taaf.diagnostics._pool_lane_events(
        [taaf.diagnostics._per_game_partial_curve(r, taaf.diagnostics.TOKEN_AXIS) for r in (one, three)],
        weights=[1.0, 3.0],
    )
    assert equal[-1] == pytest.approx(50.0)
    assert weighted[-1] == pytest.approx(75.0)


async def test_the_step_curve_survives_a_round_trip_through_the_saved_run(tmp_path: Path):
    """Intermediate scores are computed offline, so they have to be
    recoverable from what a finished run wrote — not only from the objects
    that were in memory when it ran."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game("g0")],
        solver=taaf.solver_examples.SolverFlipLR(report_tokens=100),
        job_dir=tmp_path,
    )
    await bm.run()
    live = taaf.diagnostics._per_game_partial_curve(bm.game_runs[0], taaf.diagnostics.TOKEN_AXIS)
    reloaded = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json")
    assert taaf.diagnostics._per_game_partial_curve(reloaded.game_runs[0], taaf.diagnostics.TOKEN_AXIS) == live
    # The solver changes its mind twice, so the lane must step back down.
    assert [y for _, y in live] == [0.0, 0.0, 100.0, 0.0, 100.0]


async def test_answer_sheet_renders_the_whole_task(tmp_path: Path):
    """The page that replaced the movie: what was asked, what the solver
    said, and what was right."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[ArcGame(spec_data=taaf.game_examples.flip_task("f", n_to_answer=2))],
        solver=taaf.solver_examples.SolverFlipLR(),
        job_dir=tmp_path,
    )
    await bm.run()
    sheet = tmp_path / "answers" / "gf_p0.html"
    assert sheet.is_file()
    html = sheet.read_text()
    assert "Given pairs" in html and "Answers" in html
    # One row per to-answer pair, each with both attempts, the truth, and a
    # verdict; the matching attempt is marked.
    assert html.count('alt="attempt 1"') == 2
    assert html.count('alt="attempt 2"') == 2
    assert html.count('alt="correct"') == 2
    assert html.count('<span style="color:#4caf50">correct</span>') == 2
    assert "&#10003;" in html
    # And the run page links to it rather than to a video.
    index = (tmp_path / "diagnostics.html").read_text()
    assert "answers/gf_p0.html" in index
    assert ".mp4" not in index


async def test_a_render_writes_only_the_answer_sheets_that_changed(tmp_path: Path):
    """The periodic save renders every few minutes for hours. Rewriting every
    finished game's sheet each time was most of a save on NFS and grew with
    every game that finished, so a sheet is written when it is new, when its
    run changed, or when the file is gone -- and the page still links them
    all."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game(f"g{i}") for i in range(3)],
        solver=taaf.solver_examples.SolverFlipLR(),
        job_dir=tmp_path,
    )
    await bm.run()
    page = tmp_path / "diagnostics.html"

    def sheets_written() -> list[str]:
        writes = taaf.diagnostics.render_run_html(bm, page)
        assert writes[-1].path == page, "the page is written last, after the sheets it links"
        return sorted(w.path.name for w in writes if w.path.parent.name == "answers")

    assert sheets_written() == [], "nothing changed since the run's own save"
    bm.game_runs[1].solver_analysis_html = "viewer/g1.html"
    (tmp_path / "answers" / "gg2_p0.html").unlink()
    writes = taaf.diagnostics.render_run_html(bm, page)
    assert sorted(w.path.name for w in writes if w.path.parent.name == "answers") == ["gg1_p0.html", "gg2_p0.html"]
    taaf.support.write_files(writes)
    assert sheets_written() == []
    assert all(f"answers/gg{i}_p0.html" in page.read_text() for i in range(3))


async def test_answer_sheet_says_withheld_when_it_cannot_score(tmp_path: Path):
    """An unscored sheet must not render a truth column that looks like a
    wrong answer."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[ArcGame(spec_data=taaf.game_examples.flip_task("f"))],
        solver=taaf.solver_examples.SolverFlipLR(),
        job_dir=tmp_path,
    )
    # minimal_diagnostics would skip the HTML entirely, so withhold via the
    # spec and keep diagnostics on.
    bm.games = [ArcGame(spec_data=taaf.game_examples.flip_task("f").withhold_answers())]
    await bm.run()
    html = (tmp_path / "answers" / "gf_p0.html").read_text()
    assert "withheld" in html
    assert "unscored" in html


async def test_summary_distinguishes_unscored_from_zero(tmp_path: Path):
    """A run reporting 0% would be indistinguishable from one that failed."""
    bm = taaf.benchmark.Benchmark(
        label="t",
        games=[ArcGame(spec_data=taaf.game_examples.flip_task("f").withhold_answers())],
        solver=taaf.solver_examples.SolverFlipLR(),
        job_dir=tmp_path,
    )
    await bm.run()
    text = taaf.diagnostics.run_summary_text(bm)
    assert "unscored" in text
    assert "correct=?/1" in text


def test_a_game_in_play_is_reported_at_the_answer_it_holds():
    """A periodic report is read while games are still playing. One holding an
    answer that is two-thirds right must read two-thirds in the scorers, at
    the end of its lane and in the summary, not as a zero or as unscored."""
    from taaf.game import RunSession

    game = ArcGame(spec_data=_multi_spec("p"))
    game.start_game(RunSession())
    right, wrong = Grid(np.array([[4]], np.int8)), Grid(np.array([[0]], np.int8))
    game.execute_action(Submission(attempts=((right, right), (right, wrong), (wrong, wrong))), generated_tokens=40)
    run = game.game_run
    assert run is not None and run.state == "playing"

    assert taaf.diagnostics.ARC_SCORER.of_run(run) == pytest.approx(200 / 3)
    events = taaf.diagnostics._per_game_partial_curve(run, taaf.diagnostics.TOKEN_AXIS)
    assert events[-1] == (40.0, pytest.approx(200 / 3))
    bm = taaf.benchmark.Benchmark(label="t")
    bm.game_runs = [run]
    assert "p: score=66.67, correct=2.0/3" in taaf.diagnostics.run_summary_text(bm)


async def test_comparison_html_over_two_runs(tmp_path: Path):
    """The cross-run page still builds on the new scorers and lane shape."""
    dirs: list[taaf.benchmark.Benchmark] = []
    for i, solver in enumerate((taaf.solver_examples.SolverFlipLR(), taaf.solver_examples.SolverAnalysing())):
        job = tmp_path / f"run{i}"
        bm = taaf.benchmark.Benchmark(
            label=f"r{i}",
            games=[taaf.game_examples.flip_game("f"), taaf.game_examples.constant_game("c")],
            solver=solver,
            job_dir=job,
        )
        await bm.run()
        dirs.append(taaf.benchmark.Benchmark.from_json(job / "benchmark.json"))
    out = tmp_path / "cmp"
    taaf.diagnostics.generate_comparison_html(dirs, out)
    assert (out / "index.html").is_file()
    assert (out / "per_game" / "f.html").is_file()


def _decode_png_b64(payload: str) -> np.ndarray:
    """A base64 PNG back to an (h, w, 3) uint8 array."""
    raw = matplotlib.image.imread(io.BytesIO(base64.b64decode(payload)), format="png")
    return np.rint(raw[:, :, :3] * 255).astype(np.uint8)


def test_a_rendered_grid_keeps_every_cell():
    """One pixel per cell, in the cell's own palette colour.

    The report scales these up with `image-rendering: pixelated`, so the
    PNG has to BE the grid rather than a picture of one: padding,
    resampling or sub-pixel placement costs whole rows rather than
    softening an edge. Colours come from the palette independently of
    the renderer, and edge cells carry values, because the first row and
    column are what goes missing.
    """
    expected = np.array(
        [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in taaf.grid.ARC_COLORS], dtype=np.uint8
    )
    rng = np.random.default_rng(0)
    for h, w in ((1, 1), (2, 2), (5, 7), (3, 29), (30, 30)):
        data = rng.integers(0, taaf.grid.N_COLORS, size=(h, w)).astype(np.int8)
        pixels = _decode_png_b64(taaf.diagnostics._grid_png_b64(Grid(data)))
        assert pixels.shape == (h, w, 3), f"{w}x{h} grid rendered at {pixels.shape[1]}x{pixels.shape[0]}"
        assert np.array_equal(pixels, expected[data]), f"{w}x{h} grid did not render its own cells"


def _played_input_heavy(spec: GameSpec, rights: tuple[int, ...], tokens: int = 100, uncached: int = 400):
    """``_played_sequence`` with every answer also charged ``uncached`` prompt
    tokens, so the charged and generation-only axes come apart."""
    from taaf.game import RunSession

    game = ArcGame(spec_data=spec)
    game.start_game(RunSession())
    right, wrong = Grid(np.array([[4]], np.int8)), Grid(np.array([[0]], np.int8))
    for n_right in rights:
        attempts = tuple((right, right) if i < n_right else (wrong, wrong) for i in range(len(spec.to_answer)))
        game.execute_action(Submission(attempts=attempts), generated_tokens=tokens, uncached_input_tokens=uncached)
    game.finish_game()
    assert game.game_run is not None
    return game.game_run


async def test_charging_input_moves_the_budget_but_not_the_scores() -> None:
    """``input_weight`` buys the same lane a later place on the axis.

    The curve's shape is a property of the game, so charging input must not
    move which answers stood or the score at each -- only how much spend
    each answer is recorded at, and therefore how far a fixed budget reaches.
    """
    scorer = taaf.diagnostics.ARC_SCORER
    axis = taaf.diagnostics.TOKEN_AXIS
    run = _played_input_heavy(_multi_spec(), rights=(1, 3, 2))
    plain = taaf.diagnostics._per_game_partial_curve(run, axis, scorer=scorer)
    charged = taaf.diagnostics._per_game_partial_curve(run, axis, scorer=scorer, input_weight=0.1)

    assert [y for _, y in charged] == [pytest.approx(y) for _, y in plain]
    assert all(cx >= px - 1e-9 for (cx, _), (px, _) in zip(charged, plain))
    assert charged[-1][0] > plain[-1][0], "an input-heavy lane must be charged more"
    assert charged[-1][0] == pytest.approx(taaf.diagnostics.charged_tokens(run, 0.1)), (
        "the curve must end where charged_tokens says the lane spent"
    )

    # Zero is the old behaviour exactly, which is what makes the default safe.
    assert taaf.diagnostics.charged_tokens(run, 0.0) == pytest.approx(taaf.diagnostics.total_tokens(run))
    assert taaf.diagnostics._per_game_partial_curve(run, axis, scorer=scorer, input_weight=0.0) == plain

    # The same budget reaches a strictly worse score once input is charged,
    # for a lane whose input is heavy enough to matter.
    budget = plain[-1][0] * 0.9
    assert taaf.diagnostics.lane_score_at_budget(run, budget, scorer, input_weight=0.5) <= (
        taaf.diagnostics.lane_score_at_budget(run, budget, scorer) + 1e-9
    )


async def test_input_weight_on_another_axis_is_refused() -> None:
    """``input_weight`` describes the token axis only; asked for on another
    axis it is refused rather than quietly dropped, which would read as a
    charged curve that never charged anything."""
    run = _played_input_heavy(_multi_spec(), rights=(3,))
    with pytest.raises(ValueError, match="input_weight"):
        taaf.diagnostics._per_game_partial_curve(run, taaf.diagnostics.WALLCLOCK_AXIS, input_weight=0.1)


def _bench_of(label: str, spends: tuple[int, ...]) -> taaf.benchmark.Benchmark:
    """A finished benchmark whose lanes each solved their game at a given spend."""
    bm = taaf.benchmark.Benchmark(label=label)
    bm.game_runs = [_played(_multi_spec(f"g{i}"), n_right=3, tokens=t) for i, t in enumerate(spends)]
    return bm


def _mean_spend_under(bm: taaf.benchmark.Benchmark, ceilings: dict[str, float]) -> float:
    """What this run's average game actually spends once its lanes are capped."""
    return sum(min(ceilings[r.run_id], taaf.diagnostics.total_tokens(r)) for r in bm.game_runs) / len(bm.game_runs)


#: A cheap run whose mean is dragged up by one long game, against a pricier
#: flat one. The skew is the point: it is what makes a ceiling solved over
#: both runs land above the cheap run's tail, so capping there leaves the
#: pricier run richer. Symmetric spends hide the bug — both solves agree.
_LIGHT_SPENDS = (10, 10, 10, 370)  # mean 100
_HEAVY_SPENDS = (300, 300, 300, 300)  # mean 300


def test_the_common_budget_puts_every_run_at_the_same_tokens_per_game():
    """The property the panel promises, and the one a shared ceiling breaks.

    A ceiling realizes a budget only for the spend distribution it was solved
    over, so one ceiling across a heavy and a light run equalizes their
    COMBINED mean and leaves the heavy run still reading richer — comparing
    the two at different budgets, which is what the panel exists to prevent.
    """
    heavy, light = _bench_of("heavy", _HEAVY_SPENDS), _bench_of("light", _LIGHT_SPENDS)
    budget = taaf.diagnostics._min_tokens_per_game_budget([heavy, light])
    assert budget == pytest.approx(100.0)  # the lighter run's own mean

    ceilings = taaf.diagnostics.ceilings_by_run_id([heavy, light], budget)
    assert _mean_spend_under(heavy, ceilings) == pytest.approx(budget)
    assert _mean_spend_under(light, ceilings) == pytest.approx(budget)

    # The pooled solve is what this replaced: it hits the budget only on
    # average ACROSS the runs, and the heavier one then reads well above it.
    shared = taaf.diagnostics.ceiling_for_budget([heavy, light], budget)
    pooled = {r.run_id: shared for b in (heavy, light) for r in b.game_runs}
    assert _mean_spend_under(heavy, pooled) == pytest.approx(154.0)
    assert _mean_spend_under(light, pooled) == pytest.approx(46.0)


def test_the_capped_score_reads_each_run_at_its_own_ceiling():
    """The score callable follows the lane's own run: a run already at the
    budget keeps every score, a pricier one is cut where the budget falls."""
    heavy, light = _bench_of("heavy", _HEAVY_SPENDS), _bench_of("light", _LIGHT_SPENDS)
    budget = taaf.diagnostics._min_tokens_per_game_budget([heavy, light])
    ceilings = taaf.diagnostics.ceilings_by_run_id([heavy, light], budget)
    score_fn = taaf.diagnostics._capped_score_fn(taaf.diagnostics.ARC_SCORER, ceilings)

    # The light run's mean is already the budget, so nothing is capped.
    assert [score_fn(r) for r in light.game_runs] == [100.0, 100.0, 100.0, 100.0]
    # Every heavy lane answered at 300, past its run's own ceiling of 100.
    assert [score_fn(r) for r in heavy.game_runs] == [0.0, 0.0, 0.0, 0.0]


def test_the_job_clock_curve_credits_a_queued_game_from_when_it_started():
    """Score over job wallclock is the mean score the job reported at that
    moment, so a game that waited for a slot scores nothing until it starts —
    which is what the pooled-lane curve, sliding every lane to a common origin,
    cannot show. The curve ends at the ``mean score`` of the run summary."""
    first = _played_sequence(_multi_spec(), rights=(0, 3))
    queued = copy.deepcopy(first)
    bench_start = datetime(2026, 1, 1, 12, 0, 0)
    queue_wait = 1000.0
    first.started_at = bench_start
    queued.started_at = bench_start + timedelta(seconds=queue_wait)
    for run in (first, queued):
        run.history = [
            dataclasses.replace(rec, wallclock_seconds=10.0 * i) for i, rec in enumerate(run.history, start=1)
        ]
        run.final_wallclock_seconds = 10.0 * len(run.history)

    xs, ys = taaf.diagnostics._score_vs_job_wallclock_curve([first, queued], bench_start, 2000.0)

    def at(t: float) -> float:
        return float(ys[xs.searchsorted(t, side="right") - 1])

    lane_score = taaf.diagnostics._arc_score(first)
    assert lane_score > 0.0
    assert at(0.0) == 0.0
    assert at(queue_wait - 1.0) == pytest.approx(lane_score / 2), "the queued game must not count before it starts"
    assert at(2000.0) == pytest.approx(lane_score), "both games solved; the curve ends at their mean score"
    assert xs[-1] == 2000.0, "the plateau runs out to the end of the job"
    assert at(queue_wait + 1.0) < at(2000.0), "the queued game's own answers land after its start"
