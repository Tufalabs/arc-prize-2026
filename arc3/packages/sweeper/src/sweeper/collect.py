"""Finished jobs to one tidy row per (arm, game).

The mechanical half of an analysis, so that it is tested rather than
re-improvised per experiment: what a job dir holds, how a run joins to the
arm that produced it, and how tokens are counted. Everything statistical —
regressions, plots, effect sizes — belongs to the experiment.

Pooling many jobs is the point rather than a convenience: run-level knobs
draw once per job, so their effects only appear across jobs, and a single
job's arms are a thin design on their own.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import math
import re
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import taaf.benchmark
import taaf.diagnostics
import taaf.game

from sweeper.agent import SWEEP_LINE, SweepCore
from sweeper.knobs import Design, Knob

#: Charged tokens per game the arms are compared at, unless a caller says
#: otherwise. Comparing arms where they happened to STOP is not a comparison:
#: with a per-level allowance, a configuration that wins more levels earns
#: more budget, so its terminal score is measured at a spend the others never
#: had. Every arm is therefore read at one budget instead.
#:
#: CHARGED, not generated: uncached prompt tokens count against it too, at
#: ``UNCACHED_INPUT_WEIGHT``. It is therefore NOT comparable with the 120k
#: generation-only budget the analyses before that charge were read at.
#:
#: Solved rather than derived, over sweep 8 (10 jobs, 3000 lanes): the budget
#: at which the mean score is where the generation-only 120k put it —
#: 13.618 against 13.609 — so what changes is what an arm pays for, not how
#: hard the benchmark is. The exact crossing is 149,779 and this is the round
#: number beside it, which moves the mean by 0.065%.
#:
#: An analytic 120k * (1 + r/E) would have said 156,650, 4.5% high: it assumes
#: every lane carries the sweep's mean input ratio and they do not, which is
#: why this is solved on the real curve instead.
DEFAULT_BUDGET_TOKENS = 150_000

#: Generated tokens one uncached prompt token displaces, on the deployment
#: these sweeps are read for. An arm that runs input-heavy is not spending
#: less than one that generates as much behind a warm prefix: the prefill
#: comes out of the same scheduler steps the generation does, so a budget
#: counting generation alone hands the input-heavy configuration free work —
#: which bites here, because the prompt knobs that raise score also raise
#: input.
#:
#: Measured on Kaggle (the ``throughput-test`` notebook): a fixed number of
#: requests at a fixed input size for a fixed generated length, sweeping only
#: how much of the input the prefix cache already held — so context length
#: could not ride along with the uncached count, as it does in any estimate
#: taken off live runs. Wall time is linear in
#: uncached input, and the rate is the per-generated-token cost over the
#: per-uncached-token cost. 4-, 16- and 22-way runs fit a per-step model (a
#: fixed weight read plus attention over batch x context) to within 1.6%,
#: which puts the submission's 28-way width — too wide to measure at this
#: context on that engine — at 9.4.
#:
#: It is an UPPER bound: whatever is constant per ROUND rather than per
#: generated token is still folded into the generated-token cost, so the true
#: rate is lower and uncached input costs somewhat more than this charges.
UNCACHED_INPUT_EXCHANGE_RATE = 9.4
#: The same number as the curve wants it: tokens charged per uncached input
#: token. Derived rather than written out, so the two cannot drift.
UNCACHED_INPUT_WEIGHT = 1.0 / UNCACHED_INPUT_EXCHANGE_RATE

#: Passed as ``register_games`` to score every lane's reasoning register.
#: Spelled out rather than implied by an empty filter, because "score
#: everything" is hours of block reads and should not be what a typo does.
ALL_GAMES = ("__all_games__",)

#: How much of an arm may crash before the arm is thrown away rather than
#: scored. A crash is the harness failing, not the configuration playing
#: badly — a dead engine scores zero however good the knobs were — so a few
#: rows of it would drag an arm's mean toward a number that says nothing
#: about what was swept. The threshold is not zero because a single game can
#: die for its own reasons; it is low because there is no honest way to
#: average a working arm with a broken one.
MAX_CRASHED_FRACTION = 0.05
#: Cancelled lanes tolerated per arm, as a fraction. Zero, because the
#: default reading of a cancelled lane is that the clock caught it mid-play
#: and its score is not where it would have ended. An analysis that stops
#: its own jobs deliberately knows better and raises this.
MAX_CANCELLED_FRACTION = 0.0

#: What ``SweepCore.describe`` writes, as a reader.
_SWEEP_LINE_RE = re.compile(
    rf"^{re.escape(SWEEP_LINE)}\s+(?P<batch>run|arm)(?:\s+(?P<index>\d+))?\s+seed=(?P<seed>-?\d+)\s+values=(?P<values>\{{.*\}})\s*$"
)


@dataclass(frozen=True)
class SweepRow:
    """One game played by one arm: the outcome, plus what was configured."""

    job_dir: str
    label: str
    run_id: str
    game_id: str
    pass_index: int
    arm_seed: int
    run_seed: int
    state: str
    levels_completed: int
    #: Terminal score under the scorer the rows were collected with — the
    #: engine's ARC number for the default, recomputed from the level
    #: snapshot for the variants. ``None`` when the engine never reported one
    #: (a crash), whatever the scorer: a crash is missing data under every
    #: scheme, and 0 would make an arm that breaks look like one that plays
    #: badly.
    score: float | None
    #: Score at the arm's common budget, read off this lane's
    #: score-vs-charged-tokens curve (``taaf.diagnostics.lane_score_at_budget``).
    score_at_budget: float | None
    actions: int
    #: Actions executed within the arm's common budget — counted on the same
    #: cumulative-charged-tokens axis ``score_at_budget`` is read at, so the
    #: two describe the same truncated lane.
    actions_at_budget: int
    generated_tokens: int
    uncached_input_tokens: int
    #: Generation plus uncached input at ``input_weight`` — what the budget
    #: actually spends, and what the ceiling is solved over. Carried per lane
    #: because the weight is a choice: a reader who wants the budget the
    #: sweep would have used under a different exchange rate can recompute it
    #: from this and the two columns above, without reloading the jobs.
    charged_tokens: float
    #: How input-heavy the lane ran: uncached prompt tokens per generated
    #: token, over the whole lane. ``None`` when nothing was generated (a
    #: crashed lane), where a ratio would be a division by zero posing as a
    #: measurement.
    input_per_generated: float | None
    #: How this lane's reasoning reads: 1 for the blanked register, 0 for the
    #: control one (``ullm.reasoning_register``). ``None`` when the lane was
    #: not scored, which is the default and also what a lane the
    #: ``register_games`` filter excluded gets -- distinct from ``0.5``, which
    #: means it WAS read and carried no reasoning to judge.
    register: float | None = None
    #: Every knob's value for this row: the arm's draw plus the job's
    #: run-level draw, which is constant within a job and varies across them.
    values: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for a DataFrame: knob values become columns."""
        row: dict[str, Any] = {
            "job_dir": self.job_dir,
            "label": self.label,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "pass_index": self.pass_index,
            "arm_seed": self.arm_seed,
            "run_seed": self.run_seed,
            "state": self.state,
            "levels_completed": self.levels_completed,
            "score": self.score,
            "score_at_budget": self.score_at_budget,
            "actions": self.actions,
            "actions_at_budget": self.actions_at_budget,
            "generated_tokens": self.generated_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "charged_tokens": self.charged_tokens,
            "input_per_generated": self.input_per_generated,
            "register": self.register,
        }
        row.update(self.values)
        return row


def _actions_at_budget(run: taaf.game.GameRun, budget: float, input_weight: float) -> int:
    """Actions executed within ``budget``, on the same cumulative axis
    ``taaf.diagnostics.lane_score_at_budget`` reads its curve from — uncached
    input included, at the same weight. A lane that finished under budget
    keeps its terminal count: the same extrapolation rule, with the same
    caveat that it only means anything for a lane cut off from OUTSIDE.
    """
    if budget >= taaf.diagnostics.charged_tokens(run, input_weight) - 1e-9:
        return len(run.history)
    executed = 0
    cumulative = 0.0
    for record in run.history:
        cumulative += record.generated_tokens + input_weight * record.uncached_input_tokens
        if cumulative > budget + 1e-9:
            break
        executed += 1
    return executed


def load_job(job_dir: Path | str) -> tuple[taaf.benchmark.Benchmark, SweepCore]:
    """A finished job's benchmark and the sweep agent that produced it.

    The agent comes from the solver sidecar TAAF already writes, so the
    design is recovered from the object that actually ran rather than from a
    file the sweep would have to keep in step with it.

    Intermediate states are NOT loaded. The score-vs-tokens curve needs one
    integer per action — how many levels were complete after it — and
    ``taaf.diagnostics`` derives that from ``actions_per_level`` instead of
    reading it off the per-step states. The states are grids, kept for
    thumbnails and movies; nothing in this module looks at them. Skipping
    them takes a job from ~2.8 s to ~0.5 s and a fifty-job sweep from 60 GB
    of reads to 200 MB.

    There is deliberately no check that the sidecar exists. Nothing here
    reads it, so its absence cannot skew a number — and
    ``test_a_job_scores_the_same_without_its_intermediate_states`` holds the
    rows identical with the file deleted.
    """
    path = Path(job_dir) / "benchmark.json"
    if not path.is_file():
        raise FileNotFoundError(f"no benchmark.json in {job_dir} — an unfinished or minimal-diagnostics run?")
    benchmark = taaf.benchmark.Benchmark.from_json(path, with_intermediate_states=False, with_solver=True)
    solver = benchmark.solver
    agent = getattr(solver, "agent", None)
    if not isinstance(agent, SweepCore):
        raise ValueError(
            f"{job_dir} was not played by a sweep agent (solver={type(solver).__name__}); "
            "a Kaggle submission records no solver at all — use parse_sweep_lines on its log."
        )
    return benchmark, agent


def _arm_verdict(arm_rows: list[SweepRow], max_crashed_fraction: float, max_cancelled_fraction: float) -> str | None:
    """Why this arm cannot be scored, or ``None`` if it can.

    Two ways an arm's games stop meaning what they say, and neither is the
    configuration playing badly:

    ``crashed`` — the harness failed. A dead engine still writes results, so
    the rows are present, well-formed and entirely zero. Tolerated below a
    threshold, because one game can die for its own reasons.

    ``cancelled`` — the run was cut off from outside, so the lane's score is
    where the clock caught it rather than where it would have ended. Both
    thresholds default to eliminating the arm, and both exist to be raised:
    a lane cut off AFTER it passed the scoring budget loses nothing, since
    the score is read at the budget and the play beyond it was never
    counted, and an analysis that stopped its own jobs knows which case it
    is looking at. Raising this keeps the arm and reads each cancelled lane
    at the point it was cut.
    """
    cancelled = sum(row.state == "cancelled" for row in arm_rows)
    if cancelled / len(arm_rows) > max_cancelled_fraction:
        return f"{cancelled}/{len(arm_rows)} games cancelled"
    crashed = sum(row.state == "crashed" for row in arm_rows)
    if crashed / len(arm_rows) > max_crashed_fraction:
        return f"{crashed}/{len(arm_rows)} games crashed"
    return None


def _drop_broken_arms(
    rows: list[SweepRow], max_crashed_fraction: float, max_cancelled_fraction: float
) -> list[SweepRow]:
    """Remove arms that cannot be scored, and say so where it cannot be missed.

    Silence is the failure mode worth engineering against. An analysis that
    quietly drops these rows explains nothing, and one that keeps them reports
    a configuration as bad when the harness or the clock was. So the report
    names the arm AND the directory to go read, because the next question is
    always why.
    """
    arms: dict[tuple[str, int], list[SweepRow]] = {}
    for row in rows:
        arms.setdefault((row.job_dir, row.arm_seed), []).append(row)
    doomed = {
        key: verdict
        for key, arm_rows in arms.items()
        if (verdict := _arm_verdict(arm_rows, max_crashed_fraction, max_cancelled_fraction))
    }
    if not doomed:
        return rows
    print(f"\n{SWEEP_LINE} ELIMINATED {len(doomed)} arm(s), not scored:", flush=True)
    for (job_dir, arm_seed), verdict in sorted(doomed.items()):
        print(
            f"  run_seed={arms[job_dir, arm_seed][0].run_seed} arm_seed={arm_seed}  {verdict}  "
            f"{Path(job_dir).resolve().as_uri()}/diagnostics.html",
            flush=True,
        )
    return [row for row in rows if (row.job_dir, row.arm_seed) not in doomed]


def _arm_ceiling(lanes: list[tuple[float, float]], budget_tokens: float) -> float:
    """Per-game token ceiling realizing ``budget_tokens`` over ONE arm's
    ``(spend, weight)`` lanes — TAAF's freeze-the-fleet solve, applied to an
    arm's games instead of a whole benchmark's. Zero-weight lanes are
    excluded here because ``ceiling_for_lanes`` leaves that to the caller.
    """
    return taaf.diagnostics.ceiling_for_lanes([(t, w) for t, w in lanes if w > 0], budget_tokens)


#: One loaded job, reusable across scorers: loading (unpickling benchmarks
#: WITH intermediate states) dominates an analysis session's wall clock,
#: while re-scoring loaded jobs is cheap — so an interactive caller loads
#: once with :func:`load_jobs` and re-runs :func:`rows_from_jobs` per scorer.
LoadedJob = tuple[Path, taaf.benchmark.Benchmark, SweepCore]


def load_jobs(job_dirs: Iterable[Path | str]) -> list[LoadedJob]:
    """Load every job once, for repeated re-scoring via ``rows_from_jobs``."""
    return [(Path(job_dir), *load_job(job_dir)) for job_dir in job_dirs]


def collect_rows(
    job_dirs: Iterable[Path | str],
    *,
    expect: Mapping[str, Knob] | None = None,
    budget_tokens: float = DEFAULT_BUDGET_TOKENS,
    input_weight: float = UNCACHED_INPUT_WEIGHT,
    scorer: taaf.diagnostics.Scorer = taaf.diagnostics.ARC_SCORER,
    max_crashed_fraction: float = MAX_CRASHED_FRACTION,
    max_cancelled_fraction: float = MAX_CANCELLED_FRACTION,
    register_games: Collection[str] | None = None,
    register_workers: int = 8,
) -> list[SweepRow]:
    """``rows_from_jobs`` over freshly loaded jobs — the one-shot form."""
    return rows_from_jobs(
        load_jobs(job_dirs),
        expect=expect,
        budget_tokens=budget_tokens,
        input_weight=input_weight,
        scorer=scorer,
        max_crashed_fraction=max_crashed_fraction,
        max_cancelled_fraction=max_cancelled_fraction,
        register_games=register_games,
        register_workers=register_workers,
    )


def _selected_for_register(game_id: str, names: Collection[str]) -> bool:
    """Whether ``game_id`` is one of ``names``.

    A game id is ``<name>-<hash>``, so a caller names the game ("ar25") or
    gives the id whole; matching on the separator keeps "ar25" from catching
    an "ar250" that shares its prefix.
    """
    if tuple(names) == ALL_GAMES:
        return True
    return any(game_id == name or game_id.startswith(f"{name}-") for name in names)


def _register_of_lane(lane: tuple[str, str]) -> float:
    """One lane's register score, in a worker process.

    Imported here rather than at module scope: with the default filter this
    module reads no spans and loads no classifier weights.
    """
    import platypus.reasoning_corpus

    job_dir, run_id = lane
    return platypus.reasoning_corpus.game_register(Path(job_dir) / "games" / run_id)


def _score_registers(rows: list[SweepRow], names: Collection[str], workers: int) -> list[SweepRow]:
    """``rows`` with ``register`` filled in on the lanes ``names`` selects.

    Each selected lane is a separate span read of tens of seconds, and a game
    NAME selects one lane per pass, so a four-pass job costs four reads per
    name. That is the whole reason the filter exists, and the reason the
    reads run in processes: the work is block I/O and JSON parsing.
    """
    wanted = [i for i, row in enumerate(rows) if _selected_for_register(row.game_id, names)]
    if not wanted:
        return rows
    lanes = [(rows[i].job_dir, rows[i].run_id) for i in wanted]
    if workers > 1 and len(lanes) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(lanes))) as pool:
            scores = list(pool.map(_register_of_lane, lanes))
    else:
        scores = [_register_of_lane(lane) for lane in lanes]
    out = list(rows)
    for index, score in zip(wanted, scores):
        out[index] = dataclasses.replace(out[index], register=score)
    return out


def rows_from_jobs(
    loaded: Sequence[LoadedJob],
    *,
    expect: Mapping[str, Knob] | None = None,
    budget_tokens: float = DEFAULT_BUDGET_TOKENS,
    input_weight: float = UNCACHED_INPUT_WEIGHT,
    scorer: taaf.diagnostics.Scorer = taaf.diagnostics.ARC_SCORER,
    max_crashed_fraction: float = MAX_CRASHED_FRACTION,
    max_cancelled_fraction: float = MAX_CANCELLED_FRACTION,
    register_games: Collection[str] | None = None,
    register_workers: int = 8,
) -> list[SweepRow]:
    """One row per (arm, game) across every loaded job, scored twice.

    ``score`` is where the lane ended; ``score_at_budget`` is where it stood
    after ``budget_tokens`` CHARGED tokens per game — generation plus uncached
    prompt tokens at ``input_weight``, so an arm that buys its score with a
    heavier prompt pays for it here. The second is the
    comparable one: with a per-level allowance, a configuration that wins more
    levels earns more budget, so terminal scores compare arms at spends they
    did not share.

    The budget is the **arm's ensemble**, not a per-game cap and not the
    sweep's: each arm's games share a per-game token ceiling chosen so that
    THAT arm's mean spend is the budget — one game overrunning is fine as
    long as its siblings pay for it. That mirrors deployment, where one
    submission is one configuration playing the set under its own total
    allowance. A per-game cap would undershoot (cheap lanes have nothing to
    offset them); a sweep-wide ceiling would let an expensive arm realize
    more spend than a frugal one while both are labeled with the same
    budget — and the arms it flatters most are exactly the high scorers an
    analysis would pick as winners.

    An arm is ELIMINATED rather than scored when more than
    ``max_crashed_fraction`` of its games crashed, or more than
    ``max_cancelled_fraction`` were cancelled — the harness failing and the
    clock running out respectively, neither of which is the arm playing
    badly. Every elimination is announced with the job directory to read.
    Either threshold at ``1.0`` keeps those arms, and a kept cancelled lane
    is read where it was cut.

    ``expect`` is the union of knobs the analysis wants columns for. A job
    recorded before a knob existed has no value for it, so the knob's
    declared ``missing`` fills in — which is what lets those jobs stay in the
    same regression instead of being dropped.

    ``register_games`` names the games whose reasoning register is read, and
    is OFF by default because reading one lane's reasoning costs tens of
    seconds against the milliseconds everything else here takes. Naming one
    game scores it across every arm, which is the comparison worth that
    price; :data:`ALL_GAMES` scores the lot. ``register_workers`` sizes the
    process pool those reads use — set it to 1 from inside another pool.
    """
    rows: list[SweepRow] = []
    for job_dir, benchmark, agent in loaded:
        weights = taaf.diagnostics.weights_by_game_id(benchmark)
        # Grouped by ARM SEED, not by pass. One seed may broadcast to every pass,
        # and those passes are then one configuration — which is how `by_arm`
        # pools them. Grouping by pass there would solve a separate ceiling per
        # pass, each from a fraction of the lanes, giving one arm several
        # ceilings: the invariant above, inverted.
        arm_lanes: dict[int, list[tuple[float, float]]] = {}
        for run in benchmark.game_runs:
            seed = agent._arm_seed_for(run.pass_index)  # noqa: SLF001 — the mapping IS the agent's rule
            arm_lanes.setdefault(seed, []).append(
                (taaf.diagnostics.charged_tokens(run, input_weight), weights.get(run.game_id, 1.0))
            )
        ceilings = {seed: _arm_ceiling(lanes, budget_tokens) for seed, lanes in arm_lanes.items()}
        arm_cache: dict[int, dict[str, Any]] = {}
        for run in benchmark.game_runs:
            seed = agent._arm_seed_for(run.pass_index)  # noqa: SLF001 — the mapping IS the agent's rule
            ceiling = ceilings[seed]
            if seed not in arm_cache:
                arm_cache[seed] = agent.design.arm_values_for(seed)
            values = dict(agent.run_values) | arm_cache[seed]
            if expect is not None:
                values = {name: values.get(name, expect[name].missing) for name in sorted(expect)}
            spent = taaf.diagnostics.total_tokens(run)
            uncached = taaf.diagnostics.uncached_input_tokens(run)
            rows.append(
                SweepRow(
                    job_dir=str(job_dir),
                    label=benchmark.label,
                    run_id=run.run_id,
                    game_id=run.game_id,
                    pass_index=run.pass_index,
                    arm_seed=seed,
                    run_seed=agent.run_seed,
                    state=run.state,
                    levels_completed=run.levels_completed,
                    # An infinite budget is the scorer's terminal score, so
                    # `score` follows the chosen scorer; a lane with no
                    # `final_score` stays None rather than being recomputed
                    # as 0.
                    #
                    # `final_score is None` is NOT the same as "crashed", and
                    # the difference is worth knowing before touching this: a
                    # crash lands after play, so a crashed lane carries its
                    # history, its tokens and a real score, and both fields
                    # read the score it had reached. Nulling them by STATE
                    # would drop real play and shrink the divisor, making one
                    # crashed game in an arm look like a better arm.
                    score=(
                        None
                        if run.final_score is None
                        else taaf.diagnostics.lane_score_at_budget(run, float("inf"), scorer)
                    ),
                    score_at_budget=taaf.diagnostics.lane_score_at_budget(
                        run, ceiling, scorer, input_weight=input_weight
                    ),
                    actions=sum(run.actions_per_level),
                    actions_at_budget=_actions_at_budget(run, ceiling, input_weight),
                    generated_tokens=spent,
                    uncached_input_tokens=uncached,
                    charged_tokens=taaf.diagnostics.charged_tokens(run, input_weight),
                    input_per_generated=uncached / spent if spent else None,
                    values=values,
                )
            )
    kept = _drop_broken_arms(rows, max_crashed_fraction, max_cancelled_fraction)
    # Scored after the eliminations: an arm being thrown away is not worth reading.
    return kept if register_games is None else _score_registers(kept, register_games, register_workers)


def parse_sweep_lines(text: str) -> dict[str, Any]:
    """Recover a design from a log — the only record a Kaggle submission leaves.

    Returns ``{"run_seed", "run_values", "arms": {index: {"seed", "values"}}}``.
    Pair it with :func:`verify_arms` to check the log against a local redraw
    of the same seeds, which is what catches an analysis running against a
    knob set that has moved since the submission.
    """
    run_seed: int | None = None
    run_values: dict[str, Any] = {}
    arms: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        match = _SWEEP_LINE_RE.match(line.strip())
        if match is None:
            continue
        values = json.loads(match.group("values"))
        seed = int(match.group("seed"))
        if match.group("batch") == "run":
            run_seed, run_values = seed, values
        else:
            arms[int(match.group("index"))] = {"seed": seed, "values": values}
    if run_seed is None:
        raise ValueError(f"no {SWEEP_LINE} run line in the log")
    return {"run_seed": run_seed, "run_values": run_values, "arms": arms}


def _as_logged(value: Any) -> Any:
    """A value as it survives the trip through the design line and back.

    ``describe()`` serialises with ``json.dumps(..., default=repr)``, which is
    lossy by design: a tuple returns as a list, anything unserialisable as its
    repr. Comparing a freshly drawn value against a parsed one therefore has to
    put both through the same lossy step, or a non-scalar knob reports a
    permanent phantom drift that looks exactly like the real thing.
    """
    return json.loads(json.dumps(value, default=repr))


def verify_arms(logged: Mapping[str, Any], design: Any) -> list[str]:
    """Differences between what a log recorded and what the design redraws now.

    An empty list means the analysis and the submission agree about what was
    configured. A non-empty one names the knobs that moved, which is the
    failure worth catching: the same seed drawing a different value means the
    knob set changed after the job ran, and pooling the two would compare
    configurations that were never the same.
    """
    problems: list[str] = []
    redrawn_run = {k: _as_logged(v) for k, v in design.run_values_for(logged["run_seed"]).items()}
    for name in sorted(set(redrawn_run) | set(logged["run_values"])):
        if redrawn_run.get(name) != logged["run_values"].get(name):
            problems.append(
                f"run knob {name!r}: log {logged['run_values'].get(name)!r} vs redraw {redrawn_run.get(name)!r}"
            )
    for index in sorted(logged["arms"]):
        entry = logged["arms"][index]
        redrawn = {k: _as_logged(v) for k, v in design.arm_values_for(entry["seed"]).items()}
        for name in sorted(set(redrawn) | set(entry["values"])):
            if redrawn.get(name) != entry["values"].get(name):
                problems.append(
                    f"arm {index} knob {name!r}: log {entry['values'].get(name)!r} vs redraw {redrawn.get(name)!r}"
                )
    return problems


def unit_for(design: Design, knob: str) -> str:
    """The unit one observation of ``knob`` is: ``arm`` or ``job``.

    Which batch a knob is in decides what counts as an observation of it, and
    that is the one thing an analysis cannot get wrong quietly: a run-level
    knob tested across games would count a single draw once per game and
    report an ``n`` in the hundreds for something that was decided ten times.
    """
    if knob in design.arm:
        return "arm"
    if knob in design.run:
        return "job"
    raise KeyError(f"unknown knob {knob!r}; the design declares {sorted({*design.arm, *design.run})}")


@dataclass(frozen=True)
class UnitRow:
    """Rows aggregated to one unit — one arm, or one job.

    ``score`` is the mean over whatever the unit contains, ``score_sem`` its
    standard error (``None`` for a single observation, where a spread is
    undefined rather than zero).
    """

    unit: str
    run_seed: int
    arm_seed: int | None
    values: dict[str, Any]
    score: float | None
    score_sem: float | None
    #: The comparable score: every lane read at the sweep's common budget.
    score_at_budget: float | None
    score_at_budget_sem: float | None
    actions: float
    #: Mean actions within the same common budget the score is read at.
    actions_at_budget: float
    generated_tokens: float
    #: Mean of the lanes' uncached-input-per-generated-token ratios;
    #: ``None`` when every lane crashed before generating anything.
    input_per_generated: float | None
    #: Mean reasoning register over the lanes that were scored, and how many
    #: those were. ``register_n`` is not ``n``: the register is usually read
    #: on one game where the score is read on all of them, and a reader who
    #: takes ``n`` for both compares a 4-lane number against a 100-lane one.
    register: float | None
    register_n: int
    n: int

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for a DataFrame: knob values become columns."""
        row: dict[str, Any] = {
            "unit": self.unit,
            "run_seed": self.run_seed,
            "arm_seed": self.arm_seed,
            "score": self.score,
            "score_sem": self.score_sem,
            "score_at_budget": self.score_at_budget,
            "score_at_budget_sem": self.score_at_budget_sem,
            "actions": self.actions,
            "actions_at_budget": self.actions_at_budget,
            "generated_tokens": self.generated_tokens,
            "input_per_generated": self.input_per_generated,
            "register": self.register,
            "register_n": self.register_n,
            "n": self.n,
        }
        row.update(self.values)
        return row


def _mean(values: Sequence[float]) -> float | None:
    """Mean, or ``None`` for nothing to average."""
    return sum(values) / len(values) if values else None


def _sem(values: Sequence[float]) -> float | None:
    """Standard error of the mean; ``None`` below two observations."""
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def by_arm(rows: Iterable[SweepRow]) -> list[UnitRow]:
    """One row per arm — the unit a per-arm knob was drawn at.

    Games inside an arm share that arm's single draw, so they are replicates
    of one observation rather than observations of their own.
    """
    grouped: dict[tuple[int, int], list[SweepRow]] = {}
    for row in rows:
        grouped.setdefault((row.run_seed, row.arm_seed), []).append(row)
    out: list[UnitRow] = []
    for (run_seed, arm_seed), group in sorted(grouped.items()):
        scores = [row.score for row in group if row.score is not None]
        budgeted = [row.score_at_budget for row in group if row.score_at_budget is not None]
        ratios = [row.input_per_generated for row in group if row.input_per_generated is not None]
        registers = [row.register for row in group if row.register is not None]
        out.append(
            UnitRow(
                unit="arm",
                run_seed=run_seed,
                arm_seed=arm_seed,
                values=dict(group[0].values),
                score=_mean(scores),
                score_sem=_sem(scores),
                score_at_budget=_mean(budgeted),
                score_at_budget_sem=_sem(budgeted),
                actions=sum(row.actions for row in group) / len(group),
                actions_at_budget=sum(row.actions_at_budget for row in group) / len(group),
                generated_tokens=sum(row.generated_tokens for row in group) / len(group),
                input_per_generated=_mean(ratios),
                register=_mean(registers),
                register_n=len(registers),
                n=len(group),
            )
        )
    return out


def by_job(rows: Iterable[SweepRow], design: Design) -> list[UnitRow]:
    """One row per job — the unit a run-level knob was drawn at.

    Built from the arm means rather than from the games directly, so an arm
    that played fewer games does not weigh more heavily in its job than the
    others; only the run-level knobs survive as values, because they are the
    only ones constant across the job.
    """
    grouped: dict[int, list[UnitRow]] = {}
    for arm in by_arm(rows):
        grouped.setdefault(arm.run_seed, []).append(arm)
    out: list[UnitRow] = []
    for run_seed, arms in sorted(grouped.items()):
        scores = [arm.score for arm in arms if arm.score is not None]
        budgeted = [arm.score_at_budget for arm in arms if arm.score_at_budget is not None]
        ratios = [arm.input_per_generated for arm in arms if arm.input_per_generated is not None]
        registers = [arm.register for arm in arms if arm.register is not None]
        out.append(
            UnitRow(
                unit="job",
                run_seed=run_seed,
                arm_seed=None,
                values={name: arms[0].values[name] for name in sorted(design.run) if name in arms[0].values},
                score=_mean(scores),
                score_sem=_sem(scores),
                score_at_budget=_mean(budgeted),
                score_at_budget_sem=_sem(budgeted),
                actions=sum(arm.actions for arm in arms) / len(arms),
                actions_at_budget=sum(arm.actions_at_budget for arm in arms) / len(arms),
                generated_tokens=sum(arm.generated_tokens for arm in arms) / len(arms),
                input_per_generated=_mean(ratios),
                register=_mean(registers),
                register_n=sum(arm.register_n for arm in arms),
                n=len(arms),
            )
        )
    return out


def score_by_arm(rows: Sequence[SweepRow]) -> dict[int, float]:
    """Mean score per arm seed — the aggregate an arm comparison starts from.

    Games with no score (crashed before the engine reported one) are left
    out rather than counted as zero: a crash is missing data, and scoring it
    as a failure would make an arm that crashes look like one that plays
    badly.
    """
    totals: dict[int, list[float]] = {}
    for row in rows:
        if row.score is not None:
            totals.setdefault(row.arm_seed, []).append(row.score)
    return {seed: sum(scores) / len(scores) for seed, scores in sorted(totals.items())}


__all__ = [
    "SweepRow",
    "UnitRow",
    "by_arm",
    "by_job",
    "collect_rows",
    "load_job",
    "parse_sweep_lines",
    "score_by_arm",
    "unit_for",
    "verify_arms",
]
