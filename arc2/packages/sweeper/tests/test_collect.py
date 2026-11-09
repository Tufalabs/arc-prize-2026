"""Finished jobs back to rows: the join from a game to the arm that played
it, the token tail TAAF charges outside ``history``, and the pooling of jobs
whose knob sets differ."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from test_sweep_agent import DESIGN, ScriptedAgent

import taaf.benchmark
import taaf.diagnostics
import taaf.game
from platypus import AgentSolver, RunConfig
from sweeper.agent import SweepAgent
from sweeper.collect import (
    ALL_GAMES,
    _selected_for_register,
    by_arm,
    by_job,
    collect_rows,
    load_job,
    load_jobs,
    parse_sweep_lines,
    rows_from_jobs,
    score_by_arm,
    unit_for,
    verify_arms,
)
from sweeper.knobs import Choice, Design, Field, Knob
from taaf.benchmark import Benchmark
from taaf.game_examples import flip_game


async def _run_job(
    job_dir: Path, *, run_seed: int = 3, arm_seeds: tuple[int, ...] = (11, 12, 13), n_passes: int | None = None
) -> None:
    """Play a small sweep into ``job_dir``.

    ``n_passes`` defaults to one pass per arm seed. Passing more than there are
    seeds is the broadcast shape, where one seed covers every pass.
    """
    agent = SweepAgent(
        inner=ScriptedAgent(),
        design=DESIGN,
        run_seed=run_seed,
        arm_seeds=arm_seeds,
        announce=False,
        run_config=RunConfig(process_mode="inprocess", max_concurrent=2),
    )
    games: list[taaf.game.Game] = [flip_game("a"), flip_game("b")]
    await Benchmark(
        label="collect-test",
        games=games,
        solver=AgentSolver(agent=agent),
        n_passes=len(arm_seeds) if n_passes is None else n_passes,
        job_dir=job_dir,
    ).run()


async def test_every_game_carries_the_arm_that_played_it(tmp_path: Path) -> None:
    """The join the whole analysis rests on: each row's knob values are the
    ones its own pass was configured with, recovered from the job dir alone."""
    await _run_job(tmp_path)
    rows = collect_rows([tmp_path])

    assert len(rows) == 6  # 2 games x 3 passes
    by_pass = {row.pass_index: row for row in rows}
    assert {row.arm_seed for row in rows} == {11, 12, 13}
    for pass_index, seed in enumerate((11, 12, 13)):
        assert by_pass[pass_index].arm_seed == seed
        assert by_pass[pass_index].values["cap"] == DESIGN.arm_values_for(seed)["cap"]
    # The run-level draw is constant within a job — that is what makes it a
    # between-jobs effect rather than a within-job one.
    assert len({row.values["flag"] for row in rows}) == 1


async def test_the_recorded_arm_is_the_one_that_governed_play(tmp_path: Path) -> None:
    """Rows that merely echo the design would pass the test above. This one
    ties the value to what the game did."""
    await _run_job(tmp_path)
    for row in collect_rows([tmp_path]):
        assert row.actions <= row.values["cap"]


async def test_jobs_pool_and_a_knob_added_later_falls_back_to_its_baseline(tmp_path: Path) -> None:
    """Two jobs, one knob apiece more than the other: `missing` is what keeps
    the older rows in the same regression instead of dropping them."""
    await _run_job(tmp_path / "old", run_seed=3)
    await _run_job(tmp_path / "new", run_seed=4)

    later = dict(DESIGN.arm) | {
        "extra": Knob(draw=Choice((1, 2)), apply=Field("inner.unused"), missing=-1),
        "flag": Knob(draw=Choice((True, False)), apply=Field("inner.flag"), missing=False),
    }
    rows = collect_rows([tmp_path / "old", tmp_path / "new"], expect=later)

    assert len(rows) == 12
    assert {row.values["extra"] for row in rows} == {-1}, "a knob no job ran gets its declared baseline"
    assert len({row.values["cap"] for row in rows}) > 1
    assert len({row.run_seed for row in rows}) == 2


async def test_score_by_arm_ignores_games_with_no_score(tmp_path: Path) -> None:
    """A crash is missing data. Counting it as zero would make an arm that
    breaks look like one that plays badly."""
    await _run_job(tmp_path)
    rows = collect_rows([tmp_path])
    rows_with_a_hole = [*rows[:-1], dataclasses.replace(rows[-1], score=None)]

    means = score_by_arm(rows_with_a_hole)
    assert set(means) == {11, 12, 13}
    assert all(isinstance(value, float) for value in means.values())


async def test_a_job_without_a_recorded_solver_says_so(tmp_path: Path) -> None:
    """A Kaggle submission writes no benchmark.json at all; a local run
    missing its solver sidecar is the same shape of problem, and the message
    has to point at the log-based path instead of failing obscurely."""
    await _run_job(tmp_path)
    (tmp_path / "solver.pkl").unlink()
    with pytest.raises(ValueError, match="not played by a sweep agent"):
        load_job(tmp_path)


def test_a_log_that_disagrees_with_the_current_design_is_caught() -> None:
    """The Kaggle check: the same seed drawing a different value means the
    knob set moved after the submission ran, so the two cannot be pooled."""
    agent = SweepAgent(inner=ScriptedAgent(), design=DESIGN, run_seed=3, arm_seeds=(11,), announce=False)
    logged = parse_sweep_lines(agent.describe())

    assert verify_arms(logged, DESIGN) == []

    moved = Design(run=DESIGN.run, arm={"cap": Knob(draw=Choice((999,)), apply=Field("inner.cap"))})
    problems = verify_arms(logged, moved)
    assert problems and "cap" in problems[0]


async def test_each_knob_is_aggregated_at_the_unit_it_was_drawn_at(tmp_path: Path) -> None:
    """The join that decides every p-value in an analysis. Games inside an arm
    share one draw of a per-arm knob, and every arm in a job shares one draw of
    a run-level knob — so counting games as observations of either would report
    an `n` many times larger than the number of decisions actually made."""
    await _run_job(tmp_path / "a", run_seed=3)
    await _run_job(tmp_path / "b", run_seed=4)
    rows = collect_rows([tmp_path / "a", tmp_path / "b"])

    assert unit_for(DESIGN, "cap") == "arm"
    assert unit_for(DESIGN, "flag") == "job"

    arms = by_arm(rows)
    jobs = by_job(rows, DESIGN)
    assert len(rows) == 12  # 2 games x 3 arms x 2 jobs
    assert len(arms) == 6  # the per-arm knob got six observations
    assert len(jobs) == 2  # the run-level knob got two, whatever the game count

    # A job's value is the mean of its arms, and only run-level knobs survive:
    # an arm knob has no single value at job level.
    assert set(jobs[0].values) == {"flag"}
    mine = [arm.score for arm in arms if arm.run_seed == jobs[0].run_seed and arm.score is not None]
    assert jobs[0].score == pytest.approx(sum(mine) / len(mine))
    # Two games in an arm, so a spread exists; one arm alone would have none.
    assert all(arm.n == 2 for arm in arms)


def test_unit_for_refuses_a_knob_the_design_does_not_declare() -> None:
    """An analysis asking about a knob no job ran is a typo or a stale
    notebook, and either way its answer would be silence."""
    with pytest.raises(KeyError, match="unknown knob"):
        unit_for(DESIGN, "not_a_knob")


def _charge_tokens(job_dir: Path, per_action: int) -> None:
    """Bill every action, so the lanes have a score-vs-tokens curve to read.

    The scripted agent has no LLM and therefore spends nothing, which makes
    every budget equivalent — fine for the rest of the suite, useless for
    testing a budget.
    """
    path = job_dir / "benchmark.json"
    data = json.loads(path.read_text())
    for run in data["game_runs"]:
        for record in run["history"]:
            record["generated_tokens"] = per_action
    path.write_text(json.dumps(data))


async def test_arms_are_scored_at_a_common_budget_not_where_they_stopped(tmp_path: Path) -> None:
    """The correction the whole sweep rests on. With a per-level allowance a
    configuration that wins more levels also earns more budget, so terminal
    scores compare arms at spends they never shared. Every lane is therefore
    read off its own score-vs-tokens curve at one budget."""
    await _run_job(tmp_path)
    _charge_tokens(tmp_path, per_action=10)

    generous = collect_rows([tmp_path], budget_tokens=10**9)
    frugal = collect_rows([tmp_path], budget_tokens=10)

    # A budget past every lane's end reproduces the terminal score exactly: a
    # lane that ended on its own would not have spent more.
    assert [row.score_at_budget for row in generous] == [row.score for row in generous]
    assert any(row.generated_tokens > 0 for row in generous), "nothing to truncate"
    # Below it, an arm cannot have done better, and somewhere it did worse.
    pairs = [(f.score_at_budget, g.score_at_budget) for f, g in zip(frugal, generous, strict=True)]
    assert all(low is not None and high is not None for low, high in pairs)
    assert all(low <= high for low, high in pairs if low is not None and high is not None)
    assert any(low < high for low, high in pairs if low is not None and high is not None)


async def test_the_budget_is_the_ensemble_not_a_per_game_cap(tmp_path: Path) -> None:
    """A single game overrunning is fine as long as ITS ARM averages to the
    budget. Capping each lane at the budget directly would undershoot, because
    cheap lanes spend less than it with nothing to offset them — so the cutoff
    is a ceiling computed over the arm's own games."""
    await _run_job(tmp_path)
    _charge_tokens(tmp_path, per_action=10)
    rows = collect_rows([tmp_path], budget_tokens=20)

    spends = [row.generated_tokens for row in rows]
    assert max(spends) > 20, "the fixture needs a lane that outspends the budget"
    # Lanes that finished under the budget keep their whole score, so the mean
    # spend the budget realises comes from letting the expensive ones run on.
    cheap = [row for row in rows if row.generated_tokens <= 20]
    assert cheap and all(row.score_at_budget == row.score for row in cheap)


def _crash_runs(job_dir: Path, *, pass_index: int, fraction: float) -> None:
    """Mark a share of one arm's games ``crashed``, as a dead engine does."""
    path = job_dir / "benchmark.json"
    data = json.loads(path.read_text())
    arm = [run for run in data["game_runs"] if run["pass_index"] == pass_index]
    for run in arm[: max(1, round(fraction * len(arm)))]:
        run["state"] = "crashed"
        run["final_score"] = 0.0
    path.write_text(json.dumps(data))


async def test_an_arm_that_mostly_crashed_is_eliminated_not_scored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure this exists for: an engine dies, the job still writes
    results, and every row is a well-formed zero. Scored, that arm reports its
    knobs as terrible on the strength of a broken worker."""
    await _run_job(tmp_path)
    _crash_runs(tmp_path, pass_index=0, fraction=1.0)

    rows = collect_rows([tmp_path])

    assert {row.pass_index for row in rows} == {1, 2}, "the crashed arm is gone"
    assert 11 not in score_by_arm(rows), "and gone from the statistics"
    report = capsys.readouterr().out
    assert "ELIMINATED" in report
    # The directory, because the next question is always why.
    assert tmp_path.resolve().as_uri() in report


async def test_one_cancelled_game_eliminates_its_arm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A cancelled lane stopped where the clock caught it, not where it ended.
    Kept, it would score an arm on a truncated game; kept quietly, it would do
    that invisibly — and it also leaves the arm averaged over a different set
    of games than its neighbours, so game difficulty stops cancelling out."""
    await _run_job(tmp_path)
    path = tmp_path / "benchmark.json"
    data = json.loads(path.read_text())
    one = next(run for run in data["game_runs"] if run["pass_index"] == 0)
    one["state"] = "cancelled"
    path.write_text(json.dumps(data))

    rows = collect_rows([tmp_path])

    assert {row.pass_index for row in rows} == {1, 2}
    report = capsys.readouterr().out
    assert "ELIMINATED" in report and "cancelled" in report
    assert tmp_path.resolve().as_uri() in report

    # The escape hatch, for the analysis that stopped the job itself: the arm
    # comes back, and the cut lane is read where it was cut rather than nulled.
    kept = collect_rows([tmp_path], max_cancelled_fraction=1.0)
    assert {row.pass_index for row in kept} == {0, 1, 2}
    cut = [row for row in kept if row.state == "cancelled"]
    assert cut and all(row.score_at_budget is not None for row in cut)


async def test_a_single_crashed_game_does_not_cost_the_whole_arm(tmp_path: Path) -> None:
    """Below the threshold the arm survives: one game can die for its own
    reasons, and throwing away its siblings would lose real measurements."""
    await _run_job(tmp_path, arm_seeds=(11, 12, 13))
    rows = collect_rows([tmp_path], max_crashed_fraction=0.9)

    assert {row.pass_index for row in rows} == {0, 1, 2}


async def test_the_threshold_can_be_switched_off(tmp_path: Path) -> None:
    """An analysis that wants to look at the wreckage should be able to."""
    await _run_job(tmp_path)
    _crash_runs(tmp_path, pass_index=0, fraction=1.0)

    assert len(collect_rows([tmp_path], max_crashed_fraction=1.0)) == 6


async def test_a_truncated_lane_keeps_the_levels_it_won_before_the_cut(tmp_path: Path) -> None:
    """The regression that produced a sweep whose expensive arms all scored
    near zero: loaded without task specs, every score-vs-tokens
    curve is flat, so a lane past its ceiling read 0 no matter what it had
    already answered. The fixture's pass-0 lanes answer right at 30 of a
    40-token spend, so a ceiling of 35 truncates the lane but not the win."""
    await _run_job(tmp_path)
    _charge_tokens(tmp_path, per_action=10)
    rows = [r for r in collect_rows([tmp_path], budget_tokens=35) if r.pass_index == 0]

    assert all(row.generated_tokens == 40 for row in rows), "fixture must overspend the ceiling"
    assert all(row.score_at_budget == pytest.approx(100.0) for row in rows)
    # And below the completion point the same lanes hold nothing yet.
    early = [r for r in collect_rows([tmp_path], budget_tokens=25) if r.pass_index == 0]
    assert all(row.score_at_budget == 0.0 for row in early)


async def test_a_job_without_task_specs_is_refused(tmp_path: Path) -> None:
    """Without the sidecar the curves cannot credit answers, and ``from_json``
    treats it as a silent no-op — so every truncated lane would read 0, an
    error shaped exactly like expensive arms playing badly."""
    await _run_job(tmp_path)
    (tmp_path / "game_specs.pkl").unlink()
    with pytest.raises(ValueError, match="task specs"):
        collect_rows([tmp_path])


async def test_frugal_siblings_do_not_subsidize_an_expensive_arm(tmp_path: Path) -> None:
    """The reason the ceiling is the ARM's and not the sweep's. A sweep-wide
    ceiling makes the fleet average the budget, so arms that stopped cheap
    hand their unspent budget to arms that overran — here the cheap arms
    (spends 20 and 10) would lift a shared ceiling to 36, past the expensive
    arm's level win at 30. That arm's own ceiling is the budget itself, 22,
    and its win does not survive a cut it could not afford."""
    await _run_job(tmp_path)
    _charge_tokens(tmp_path, per_action=10)
    rows = collect_rows([tmp_path], budget_tokens=22)

    expensive = [r for r in rows if r.pass_index == 0]
    assert all((row.score or 0) > 0 for row in expensive), "fixture: the arm did win the level"
    assert all(row.score_at_budget == 0.0 for row in expensive)
    # The cheap arms are untouched either way: they ended on their own, and
    # more budget would not have changed what they did.
    assert all(row.score_at_budget == row.score for row in rows if row.pass_index != 0)


async def test_a_single_arm_agrees_with_taafs_own_solve(tmp_path: Path) -> None:
    """One arm spanning the whole job is the case where the per-arm solve and
    ``taaf.diagnostics.ceiling_for_budget`` answer the identical question, so
    they must agree exactly — the pin that keeps the local solve from
    drifting off TAAF's."""
    await _run_job(tmp_path, arm_seeds=(11,))
    _charge_tokens(tmp_path, per_action=10)
    rows = collect_rows([tmp_path], budget_tokens=35)

    benchmark = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json", with_game_specs=True, with_solver=False)
    ceiling = taaf.diagnostics.ceiling_for_budget([benchmark], 35)
    expected = {
        run.run_id: taaf.diagnostics.lane_score_at_budget(run, ceiling, taaf.diagnostics.ARC_SCORER)
        for run in benchmark.game_runs
    }
    assert {row.run_id: row.score_at_budget for row in rows} == expected


async def test_a_broadcast_seed_gets_one_ceiling_for_all_its_passes(tmp_path: Path) -> None:
    """One arm seed over many passes is ONE configuration — `_arm_seed_for`
    broadcasts it and `by_arm` pools it — so it must be read at one ceiling.
    Solving per pass instead would give the same arm several ceilings, each
    from a fraction of its lanes, which is the per-arm invariant inverted."""
    await _run_job(tmp_path, arm_seeds=(11,), n_passes=3)
    # Charged UNEVENLY by pass. Equal spend makes every per-pass solve land on
    # the same ceiling, so a split solve would pass unnoticed; the passes have
    # to disagree for the two groupings to differ at all.
    path = tmp_path / "benchmark.json"
    data = json.loads(path.read_text())
    for run in data["game_runs"]:
        for record in run["history"]:
            record["generated_tokens"] = 4 + 8 * run["pass_index"]
    path.write_text(json.dumps(data))

    rows = collect_rows([tmp_path], budget_tokens=35)
    assert len({row.pass_index for row in rows}) == 3, "fixture: the seed did broadcast"

    # One arm spanning the whole job asks TAAF's own question, so it must give
    # TAAF's own answer — which it cannot if the solve is split by pass.
    benchmark = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json", with_game_specs=True, with_solver=False)
    ceiling = taaf.diagnostics.ceiling_for_budget([benchmark], 35)
    expected = {
        run.run_id: taaf.diagnostics.lane_score_at_budget(run, ceiling, taaf.diagnostics.ARC_SCORER)
        for run in benchmark.game_runs
    }
    assert {row.run_id: row.score_at_budget for row in rows} == expected


async def test_the_scorer_reaches_both_scores(tmp_path: Path) -> None:
    """Switching scorer must switch the TERMINAL score too, not only the
    budgeted one — a notebook comparing `score` to `score_at_budget` under
    the solved scorer would otherwise mix two scales in one plot. A crash
    stays None under every scheme: missing data, not a zero."""
    await _run_job(tmp_path)
    solved = collect_rows([tmp_path], scorer=taaf.diagnostics.SOLVED_SCORER)
    # One pair per task: solved is the level won, on both scales.
    assert all(row.score == row.levels_completed for row in solved)
    assert all(row.score_at_budget == row.score for row in solved)
    assert {row.score for row in solved} == {0.0, 1.0}


async def test_actions_at_budget_counts_on_the_score_curves_axis(tmp_path: Path) -> None:
    """The action count is read at the same per-arm ceiling as
    ``score_at_budget``: a lane that finished under budget keeps its terminal
    count, one the ceiling truncates counts only the actions whose cumulative
    generated tokens fit."""
    await _run_job(tmp_path, arm_seeds=(11,))
    _charge_tokens(tmp_path, per_action=10)

    generous = collect_rows([tmp_path], budget_tokens=10_000)
    assert all(row.actions_at_budget == row.actions for row in generous)

    tight = collect_rows([tmp_path], budget_tokens=35)
    benchmark = taaf.benchmark.Benchmark.from_json(tmp_path / "benchmark.json", with_game_specs=True, with_solver=False)
    ceiling = taaf.diagnostics.ceiling_for_budget([benchmark], 35)
    by_id = {run.run_id: run for run in benchmark.game_runs}
    for row in tight:
        run = by_id[row.run_id]
        expected = len(run.history) if ceiling >= 10 * len(run.history) else int(ceiling // 10)
        assert row.actions_at_budget == expected
        assert row.actions_at_budget <= row.actions


async def test_input_per_generated_is_the_lanes_input_heaviness(tmp_path: Path) -> None:
    """The ratio is uncached input per generated token over the whole lane —
    and None, not zero, when nothing was generated: a lane that spent nothing
    has no input-heaviness, and 0 would read as maximally output-heavy."""
    await _run_job(tmp_path, arm_seeds=(11,))
    unbilled = collect_rows([tmp_path])
    assert {row.input_per_generated for row in unbilled} == {None}

    path = tmp_path / "benchmark.json"
    data = json.loads(path.read_text())
    for run in data["game_runs"]:
        for record in run["history"]:
            record["generated_tokens"] = 10
            record["uncached_input_tokens"] = 25
    path.write_text(json.dumps(data))
    billed = collect_rows([tmp_path])
    assert all(row.input_per_generated == pytest.approx(2.5) for row in billed)


async def test_loaded_jobs_rescore_without_reloading(tmp_path: Path) -> None:
    """The interactive split: one load serves every scorer. The rows must
    match what a fresh one-shot collect produces, or the cheap path would be
    a different analysis."""
    await _run_job(tmp_path)
    loaded = load_jobs([tmp_path])
    arc = rows_from_jobs(loaded)
    solved = rows_from_jobs(loaded, scorer=taaf.diagnostics.SOLVED_SCORER)

    assert [row.run_id for row in arc] == [row.run_id for row in solved]
    assert all(row.score == row.levels_completed for row in solved)
    fresh = collect_rows([tmp_path])
    assert [row.to_dict() for row in fresh] == [row.to_dict() for row in arc]


async def test_a_lane_that_gave_up_unanswered_is_a_zero_not_missing_data(tmp_path: Path) -> None:
    """The configuration was asked and did not answer. Treating that as
    missing data would score an arm on the games it chose to answer — half a
    sweep's lanes, on a task set where giving up is how most games end. Both
    columns read the same population, which is the point: a run always
    carries a score, so neither has to reinterpret its absence."""
    silent = Design(run=DESIGN.run, arm={"cap": Knob(draw=Choice((0,)), apply=Field("inner.cap"), missing=100)})
    agent = SweepAgent(
        inner=ScriptedAgent(),
        design=silent,
        run_seed=3,
        arm_seeds=(11, 12),
        announce=False,
        run_config=RunConfig(process_mode="inprocess", max_concurrent=2),
    )
    games: list[taaf.game.Game] = [flip_game("a"), flip_game("b")]
    await Benchmark(label="silent", games=games, solver=AgentSolver(agent=agent), n_passes=2, job_dir=tmp_path).run()

    rows = collect_rows([tmp_path])
    assert len(rows) == 4 and all(row.state == "gave_up" for row in rows)
    assert all(row.score == 0.0 for row in rows), [row.score for row in rows]
    assert all(row.score_at_budget == 0.0 for row in rows)
    assert all(arm.score == 0.0 and arm.n == 2 for arm in by_arm(rows))
    assert score_by_arm(rows) == {11: 0.0, 12: 0.0}

    # Crashing does not withdraw a score. A crash lands after play, so the
    # lane keeps what it had reached — here 0.0, because it answered nothing
    # — and the state is what an analysis filters on. Nulling by state would
    # shrink the divisor and make one crashed game look like a better arm.
    path = tmp_path / "benchmark.json"
    data = json.loads(path.read_text())
    next(run for run in data["game_runs"] if run["pass_index"] == 0)["state"] = "crashed"
    path.write_text(json.dumps(data))
    crashed = [row for row in collect_rows([tmp_path], max_crashed_fraction=1.0) if row.state == "crashed"]
    assert len(crashed) == 1 and crashed[0].score == 0.0 and crashed[0].score_at_budget == 0.0


async def test_an_input_heavy_lane_pays_for_its_prompt(tmp_path: Path) -> None:
    """The reason the budget charges input at all.

    Two lanes generating identically, one of them reading a far heavier
    prompt, are not spending the same: the prefill comes out of the same
    scheduler steps the generation does. Charged at ``input_weight`` the
    heavy lane reaches the budget sooner and is read earlier on its own
    curve, so a prompt knob that buys score with tokens cannot show up as
    free. At ``input_weight=0`` the two are indistinguishable, which is what
    the old budget did and what the flag still reproduces.
    """
    await _run_job(tmp_path)
    _charge_tokens(tmp_path, per_action=10)
    path = tmp_path / "benchmark.json"
    data = json.loads(path.read_text())
    for run in data["game_runs"]:
        for record in run["history"]:
            record["uncached_input_tokens"] = 100
    path.write_text(json.dumps(data))

    plain = collect_rows([tmp_path], budget_tokens=35, input_weight=0.0)
    charged = collect_rows([tmp_path], budget_tokens=35, input_weight=0.5)

    by_run = {row.run_id: row for row in plain}
    dropped = False
    for row in charged:
        twin = by_run[row.run_id]
        assert row.generated_tokens == twin.generated_tokens, "the charge must not touch what was generated"
        assert row.uncached_input_tokens == twin.uncached_input_tokens
        assert row.charged_tokens > twin.charged_tokens, "input must be charged into the spend"
        assert row.charged_tokens == pytest.approx(twin.generated_tokens + 0.5 * twin.uncached_input_tokens)
        assert row.actions_at_budget <= twin.actions_at_budget, (
            "a charged lane reaches the same budget after fewer actions"
        )
        assert row.score_at_budget is not None and twin.score_at_budget is not None, "the fixture's lanes all score"
        assert row.score_at_budget <= twin.score_at_budget + 1e-9
        dropped = dropped or row.score_at_budget < twin.score_at_budget

    assert dropped, "with input this heavy the charge has to change at least one lane's reading"


async def test_the_reasoning_register_is_off_unless_a_game_is_named(tmp_path: Path) -> None:
    """Default-off, and off means unread: no lane carries a score.

    ``None`` and ``0.5`` are different answers -- not scored, versus scored
    and carrying nothing to judge -- so the default must produce the first.
    """
    await _run_job(tmp_path)
    assert {row.register for row in collect_rows([tmp_path])} == {None}
    assert {arm.register for arm in by_arm(collect_rows([tmp_path]))} == {None}
    assert {arm.register_n for arm in by_arm(collect_rows([tmp_path]))} == {0}


async def test_naming_one_game_scores_it_in_every_arm_and_nothing_else(tmp_path: Path) -> None:
    """The point of the filter: one game across the arms, at one lane's cost each.

    These lanes have no LLM spans to read, so the score is the classifier's
    no-evidence answer; what is under test is which lanes were read at all.
    """
    await _run_job(tmp_path)
    plain = collect_rows([tmp_path])
    chosen = plain[0].game_id
    others = {row.game_id for row in plain} - {chosen}
    assert others, "the fixture plays more than one game, or this proves nothing"

    rows = collect_rows([tmp_path], register_games=(chosen,), register_workers=1)
    scored = {row.game_id for row in rows if row.register is not None}
    assert scored == {chosen}
    assert all(row.register == 0.5 for row in rows if row.game_id == chosen)
    # One lane per pass, so naming one game costs a read for each arm that played it.
    arms = {arm.arm_seed: arm for arm in by_arm(rows)}
    assert all(arm.register == 0.5 and arm.register_n == 1 for arm in arms.values())
    assert all(arm.n > arm.register_n for arm in arms.values()), "n describes the score, register_n the register"


async def test_all_games_scores_every_lane(tmp_path: Path) -> None:
    """The sentinel is the only way to ask for everything, so it has to work."""
    await _run_job(tmp_path)
    rows = collect_rows([tmp_path], register_games=ALL_GAMES, register_workers=1)
    assert all(row.register == 0.5 for row in rows)


def test_a_game_name_matches_its_lanes_and_not_a_longer_sibling() -> None:
    """``ar25`` is the name of ``ar25-0c556536`` and nothing else."""
    assert _selected_for_register("ar25-0c556536", ("ar25",))
    assert _selected_for_register("ar25-0c556536", ("ar25-0c556536",))
    assert not _selected_for_register("ar250-0c556536", ("ar25",))
    assert not _selected_for_register("bp35-0a0ad940", ("ar25",))
    assert _selected_for_register("anything-at-all", ALL_GAMES)
