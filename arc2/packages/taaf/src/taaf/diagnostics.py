"""Per-benchmark and cross-benchmark diagnostics (R4.*).

Public surface:

- ``generate_run_html`` — per-run HTML (``render_run_html`` renders it
  without writing it)
- ``generate_run_summary_txt`` — text summary
- ``generate_comparison_html`` — cross-run comparison
- ``regenerate_run_diagnostics`` / ``regenerate_comparison_diagnostics``
  — load + regen from a finished run dir

PNGs are embedded inline as base64. Each per-(game, pass) page is a static
answer sheet: what the task asked, what the solver answered, what was right.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import statistics
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats

import taaf.benchmark
import taaf.game
import taaf.grid
import taaf.support

matplotlib.use("Agg")  # headless rendering

# scipy.stats.ttest_rel emits "Precision loss occurred" when paired
# differences have near-zero variance (common with small n_games). The
# p-value is still meaningful for our use; the warning is noise. Inner
# ``catch_warnings`` blocks don't suppress reliably under ipykernel /
# pytest, so install a global filter at import time.
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred",
    category=RuntimeWarning,
    module=r"scipy\.stats.*",
)


# --- Style constants --------------------------------------------------------

BG_COLOR = "#1e1e1e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#3a3a3a"
ACCENT_COLORS = [
    "#9bd1ff",
    "#ffb47a",
    "#a8e6a1",
    "#e6a1d3",
    "#f0d878",
    "#ff9b9b",
    "#b39bff",
    "#d4e67a",
    "#9be6e6",
    "#ff9bc8",
    "#9bb4ff",
    "#9be6cb",
]
# Distinct palette for the comparison page so per-run vs cross-run
# plots are visually distinguishable side-by-side.
RUN_COLORS = [
    "#ff6b9d",
    "#7dd3fc",
    "#fbbf24",
    "#a78bfa",
    "#34d399",
    "#f87171",
    "#fb923c",
    "#84cc16",
    "#2dd4bf",
    "#818cf8",
    "#e879f9",
    "#4ade80",
]
# Muted colors for ``GameRun.state`` cells — low saturation so the
# column reads as metadata, not competing with the curve palette.
_STATE_COLORS: dict[str, str] = {
    "won": "#6db04e",  # muted green
    "gave_up": "#8a8a8a",  # gray
    "cancelled": "#d09c4d",  # muted amber
    "crashed": "#d05050",  # muted red
    "playing": "#5a92c8",  # muted blue (shouldn't appear after teardown)
    "not_started": "#666",
}


def _state_cell_html(state: str) -> str:
    """A table cell colouring a run state."""
    return f'<td style="color: {_STATE_COLORS.get(state, "#888")}">{escape(state)}</td>'


THUMB_DEFAULT_PX = 32
THUMB_FINAL_PX = 64


# --- Score variants ---------------------------------------------------------
# Two scorers feed the per-run and comparison HTML's tab widget. Both read a
# finished run's own result rather than deriving anything from levels: with
# one level per game every level-shaped scorer collapses onto "did any pair
# come back right", which is what ``won`` already says.


@dataclass(frozen=True)
class Scorer:
    """One scoring scheme, evaluated on a ``GameRun`` finished or in play.

    Fields:

    - ``key``: url-safe id used in filenames / cache keys.
    - ``label``: display name in nav links and plot titles.
    - ``description``: short HTML paragraph under the page's score nav.
    - ``y_axis_label``: y-axis label on the cross-game pooled curves.
    - ``path_suffix``: filename suffix for this variant's HTML (``""`` for ARC).
    - ``of_result``: the scheme itself, over one ``GameResult``. This is what
      lets a lane's *earlier* answers be scored the same way as its last,
      from the recorded history.
    - ``of_run``: the score of the run as it stands (``GameRun.current_result``),
      whatever its state. Returns 0.0 where nothing can be scored; callers
      that must tell *unscored* from *zero* read ``run.current_score() is
      None`` instead.
    """

    key: str
    label: str
    description: str
    y_axis_label: str
    path_suffix: str
    of_result: Callable[[taaf.game.GameResult], float]
    of_run: Callable[[taaf.game.GameRun], float]


def _arc_result_score(result: taaf.game.GameResult) -> float:
    """Percentage of to-answer pairs this answer got right."""
    return result.score or 0.0 if result.scored else 0.0


def _arc_score(run: taaf.game.GameRun) -> float:
    """Percentage of to-answer pairs the run has right as it stands, 0.0 where
    that cannot be computed."""
    score = run.current_score()
    return 0.0 if score is None else score


def _solved_result_score(result: taaf.game.GameResult) -> float:
    """1 when every pair of this answer came back right, else 0."""
    if not result.scored:
        return 0.0
    return 1.0 if result.n_correct == len(result.pairs) else 0.0


def _solved_score(run: taaf.game.GameRun) -> float:
    """1 when every pair came back right, else 0.

    ARC-AGI-2's own task-solved criterion, which ``GameRun.won`` no longer
    carries: at one level per game the level is beaten by a single correct
    pair, so "solved" needs a scorer of its own.
    """
    result = run.current_result()
    return 0.0 if result is None else _solved_result_score(result)


ARC_SCORER = Scorer(
    key="arc",
    label="Official ARC",
    description=(
        "Percentage of test outputs answered correctly, taking each output "
        "independently and either attempt for each. Averaged over games with "
        "equal weight this is <code>arc-agi-benchmarking</code>'s score; "
        "weight each game by its number of test outputs and it is Kaggle's."
    ),
    y_axis_label="average score per game",
    path_suffix="",
    of_result=_arc_result_score,
    of_run=_arc_score,
)
SOLVED_SCORER = Scorer(
    key="solved",
    label="Tasks solved",
    description=(
        "All-or-nothing: 1 only when <em>every</em> test output of the task is "
        "right, which is ARC-AGI-2's own definition of solving a task. It "
        "differs from the score above on any task with more than one test "
        "output &mdash; 37.5% of the public evaluation set."
    ),
    y_axis_label="fraction of tasks solved",
    path_suffix="_solved",
    of_result=_solved_result_score,
    of_run=_solved_score,
)

SCORERS: list[Scorer] = [ARC_SCORER, SOLVED_SCORER]


def _score_for(run: taaf.game.GameRun, scorer: Scorer) -> float:
    """Terminal score for ``run`` under ``scorer``."""
    return scorer.of_run(run)


def _resolve_score_fn(
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None,
) -> Callable[[taaf.game.GameRun], float]:
    """Per-run score callable used by the stats helpers. ``None`` ⇒ the
    terminal score (run evaluated where it ended). The common-budget
    variant passes a function that reads each run's score-vs-tokens curve
    at a shared per-game token budget instead."""
    if score_fn is not None:
        return score_fn

    def _terminal(run: taaf.game.GameRun) -> float:
        return _score_for(run, scorer)

    return _terminal


def _scorer_tabs_html(panels: list[tuple[Scorer, str]]) -> str:
    """CSS-only folder-tab widget. ``panels`` is a list of
    ``(scorer, panel_inner_html)`` tuples; the first is the default-active
    tab. With a single panel (in-flight: ARC only), the tab chrome is
    skipped and the panel's HTML is returned bare so the page lays out as
    if there were no tab UI.
    """
    if not panels:
        return ""
    if len(panels) == 1:
        return panels[0][1]
    inputs: list[str] = []
    labels: list[str] = []
    panel_divs: list[str] = []
    for i, (scorer, inner_html) in enumerate(panels):
        checked = " checked" if i == 0 else ""
        rid = f"scorer-{scorer.key}"
        inputs.append(f'<input type="radio" name="scorer" id="{rid}"{checked}>')
        labels.append(f'<label for="{rid}" class="tab-label">{escape(scorer.label)}</label>')
        panel_divs.append(
            f'<div class="tab-panel" id="panel-{scorer.key}">'
            f'<p class="scorer-desc">{scorer.description}</p>'
            f"{inner_html}"
            "</div>"
        )
    return (
        '<div class="scorer-tabs">'
        + "".join(inputs)
        + '<div class="tab-row">'
        + "".join(labels)
        + "</div>"
        + "".join(panel_divs)
        + "</div>"
    )


def _budget_scorer_tabs_html(
    scorer_panels: list[tuple[Scorer, dict[str, str]]],
    *,
    budget_labels: list[tuple[str, str]],
) -> str:
    """Two-level CSS-only tab widget: an outer row of budget tabs
    above an inner row of score-variant tabs. A panel is visible only
    when its budget tab and its scorer tab are both selected — so the
    statistics are shown at each run's endpoint and at a shared per-game
    token budget, for each scorer.

    ``budget_labels`` is ``[(key, label), …]`` (keys ``full`` / ``capped``,
    matching the hardcoded visibility CSS); ``scorer_panels`` maps each
    scorer to its ``{budget_key: inner_html}``. With a single budget mode
    the budget dimension is dropped — the scorer-only widget is used.
    """
    if len(budget_labels) < 2:
        only = budget_labels[0][0]
        return _scorer_tabs_html([(s, by[only]) for s, by in scorer_panels])
    inputs: list[str] = []
    for i, (bk, _) in enumerate(budget_labels):
        inputs.append(f'<input type="radio" name="bs-budget" id="bs-b-{bk}"{" checked" if i == 0 else ""}>')
    for i, (scorer, _) in enumerate(scorer_panels):
        inputs.append(f'<input type="radio" name="bs-scorer" id="bs-s-{scorer.key}"{" checked" if i == 0 else ""}>')
    budget_row = "".join(f'<label for="bs-b-{bk}" class="tab-label">{escape(bl)}</label>' for bk, bl in budget_labels)
    scorer_row = "".join(
        f'<label for="bs-s-{s.key}" class="tab-label">{escape(s.label)}</label>' for s, _ in scorer_panels
    )
    panel_divs: list[str] = []
    for scorer, by_budget in scorer_panels:
        for bk, _ in budget_labels:
            panel_divs.append(
                f'<div class="tab-panel" id="bs-p-{bk}-{scorer.key}">'
                f'<p class="scorer-desc">{scorer.description}</p>'
                f"{by_budget[bk]}"
                "</div>"
            )
    return (
        '<div class="bs-tabs">'
        + "".join(inputs)
        + f'<div class="tab-row budget-row">{budget_row}</div>'
        + f'<div class="tab-row scorer-row">{scorer_row}</div>'
        + "".join(panel_divs)
        + "</div>"
    )


_VARIANT_REMAINDER_NOTE = (
    '<p class="design-note">Sections below this point — per-game cards, '
    "per-game drill-downs, and the per-pass-per-game table — always render "
    "under the official ARC-AGI score, independent of the variant chosen above."
    "</p>"
)


def _weights_banner_html(benchmark: taaf.benchmark.Benchmark) -> str:
    """Yellow banner shown when ``Benchmark.game_weights`` is set.
    Summarises with totals + weight-0 exclusion count — a full per-game
    list would dominate the page on large benchmarks.
    """
    if benchmark.game_weights is None:
        return ""
    weights = weights_by_game_id(benchmark)
    if not weights:
        return ""
    n_total = len(weights)
    n_excluded = sum(1 for w in weights.values() if w == 0)
    excl_note = f" {n_excluded} game{'s' if n_excluded != 1 else ''} excluded (weight 0)." if n_excluded else ""
    return (
        '<div class="weights-banner">'
        "<p><strong>Game weights applied.</strong> The pooled curves, "
        "per-pass mean, head-to-head means, and t-tests below weight each "
        "game per <code>Benchmark.game_weights</code> "
        f"({n_total} game{'s' if n_total != 1 else ''} total.{excl_note}) "
        "Per-game cards and the per-pass-per-game table at the bottom "
        "remain unweighted.</p>"
        "</div>"
    )


def _comparison_weights_banner_html(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
) -> str:
    """Yellow banner for the comparison page. ``generate_comparison_html``
    has harmonized all benchmarks to share weights, so the banner
    describes the single shared weighting.
    """
    del labels
    if not any(b.game_weights is not None for b in benchmarks):
        return ""
    weights = weights_by_game_id(benchmarks[0])
    if not weights:
        return ""
    n_total = len(weights)
    n_excluded = sum(1 for w in weights.values() if w == 0)
    excl_note = f" {n_excluded} game{'s' if n_excluded != 1 else ''} excluded (weight 0)." if n_excluded else ""
    return (
        '<div class="weights-banner">'
        "<p><strong>Game weights applied.</strong> Cross-game aggregation "
        "(pooled curves, per-run mean, t-tests, head-to-head means) uses "
        f"the shared <code>Benchmark.game_weights</code> ({n_total} game"
        f"{'s' if n_total != 1 else ''} total.{excl_note}) "
        "Per-game grid stays unweighted.</p>"
        "</div>"
    )


# --- Data extraction --------------------------------------------------------


def total_tokens(run: taaf.game.GameRun) -> int:
    """Generation charged to this run, no-move tail included.

    Public because an analysis outside TAAF needs the same number the curves
    and grand totals use; a second implementation is a second thing to keep in
    agreement.
    """
    # Folding the no-move give-up/cancel tokens in at this single
    # primitive means every grand-total consumer and every score-vs-tokens
    # curve (whose terminal x is this total) picks them up unchanged.
    return sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens


def _tokens_at_record(run: taaf.game.GameRun, index: int) -> float:
    """Generation charged to this run by the time answer ``index`` stood.

    ``ActionRecord.generated_tokens`` is the spend *since the previous*
    answer, so this is a running sum — the finish tail is deliberately absent,
    since it was not yet spent when this answer went in.
    """
    return float(sum(rec.generated_tokens for rec in run.history[: index + 1]))


def _charged_tokens_at_record(run: taaf.game.GameRun, index: int, input_weight: float) -> float:
    """:func:`_tokens_at_record` with uncached input charged at ``input_weight``
    — the per-answer running sum on the same axis :func:`charged_tokens`
    totals, so a budget read against one is read against the other."""
    return float(
        sum(rec.generated_tokens + input_weight * rec.uncached_input_tokens for rec in run.history[: index + 1])
    )


def _wallclock_at_record(run: taaf.game.GameRun, index: int) -> float:
    """Seconds elapsed when answer ``index`` was submitted. Already cumulative
    — it shares ``final_wallclock_seconds``'s since-start_game reference."""
    return run.history[index].wallclock_seconds


def uncached_input_tokens(run: taaf.game.GameRun) -> int:
    """Prompt tokens this run had to have computed, no-move tail included.

    The input-side mirror of :func:`total_tokens`, and public for the same
    reason: an analysis that charges input as well as generation needs the
    number the curves charge, not a second sum that can drift from it.
    """
    return sum(rec.uncached_input_tokens for rec in run.history) + run.final_uncached_input_tokens


def charged_tokens(run: taaf.game.GameRun, input_weight: float = 0.0) -> float:
    """What a lane spent, charging uncached input at ``input_weight``.

    ``input_weight`` is the reciprocal of an exchange rate: how many generated
    tokens one uncached prompt token displaces. It exists because an arm that
    ends up input-heavy is not spending less than one that generates the same
    amount with a warm prefix -- the prefill comes out of the same steps the
    generation does -- so comparing arms on generation alone hands the
    input-heavy configuration free work.

    Zero is generation-only, which is what every curve charged before this
    existed and what the default keeps them charging. The number is a property
    of the deployment being compared, not of TAAF, so it is the caller's to
    supply.
    """
    total = float(total_tokens(run))
    if input_weight:
        total += input_weight * uncached_input_tokens(run)
    return total


def _total_wallclock(run: taaf.game.GameRun) -> float:
    """Seconds this run occupied, no-move give-up tail included."""
    # Fold in the no-move give-up/cancel turn (mirrors total_tokens).
    # final_wallclock_seconds and ActionRecord.wallclock_seconds share the
    # same monotonic-since-start_game reference; finish_game() stamps final
    # after the last action, so max() picks the finish time once finalized
    # and the last action while in-flight (final defaults to 0.0).
    last = run.history[-1].wallclock_seconds if run.history else 0.0
    return max(last, run.final_wallclock_seconds)


@dataclass(frozen=True)
class CostAxis:
    """What a lane's spend is measured in, at the end *and* along the way.

    The pair is the point. A lane steps wherever a solver replaced its
    answer, not only at its end, so plotting one needs the cost at each
    answer in the same unit as the total. Keeping both halves in one
    object is what stops a curve reading its x from one measure and its knots
    from another.
    """

    total: Callable[[taaf.game.GameRun], float]
    at_record: Callable[[taaf.game.GameRun, int], float]


TOKEN_AXIS = CostAxis(total=lambda run: float(total_tokens(run)), at_record=_tokens_at_record)
WALLCLOCK_AXIS = CostAxis(total=_total_wallclock, at_record=_wallclock_at_record)


def _benchmark_elapsed_seconds(benchmark: taaf.benchmark.Benchmark) -> float | None:
    """Job wallclock so far, or None before the run started."""
    if benchmark.start_time is None:
        return None
    end_time = benchmark.end_time
    if end_time is None:
        if benchmark.start_time.tzinfo is None:
            end_time = datetime.now()
        else:
            end_time = datetime.now(benchmark.start_time.tzinfo)
    return max((end_time - benchmark.start_time).total_seconds(), 0.0)


def _count_actions(run: taaf.game.GameRun) -> int:
    """How many answers the solver submitted."""
    return len(run.history)


def _n_to_answer(run: taaf.game.GameRun) -> int:
    """How many pairs this task asked the solver to answer."""
    if run.result is not None:
        return len(run.result.pairs)
    return len(run.spec.to_answer) if run.spec is not None else 0


def _correct_cell(run: taaf.game.GameRun) -> str:
    """The correct/total cell, as the run stands. Unscored reads ``?/n``, never
    ``0/n`` — a withheld run is not a run that got everything wrong — and no
    answer at all reads ``—/n``."""
    n = _n_to_answer(run)
    if not run.answered:
        return f"&mdash;/{n}"
    result = run.current_result()
    if result is None or not result.scored:
        return f"?/{n}"
    return f"{result.n_correct}/{n}"


def _game_ids(benchmark: taaf.benchmark.Benchmark) -> list[str]:
    """Unique game_ids in alphabetical order — stable display order
    independent of the solver's dispatch order. Use ``benchmark.game_runs``
    directly if you need play order.
    """
    return sorted({r.game_id for r in benchmark.game_runs})


def _pass_runs(benchmark: taaf.benchmark.Benchmark, pass_idx: int) -> list[taaf.game.GameRun]:
    """All runs in the given pass, one per game, in benchmark game order
    (``GameRun.pass_index``)."""
    return [r for r in benchmark.game_runs if r.pass_index == pass_idx]


def _game_runs_by_id(benchmark: taaf.benchmark.Benchmark, game_id: str) -> list[taaf.game.GameRun]:
    """Every pass of one game, in pass order."""
    return [r for r in benchmark.game_runs if r.game_id == game_id]


def weights_by_game_id(benchmark: taaf.benchmark.Benchmark) -> dict[str, float]:
    """Resolve ``Benchmark.game_weights`` (list parallel to ``games``)
    into a ``{game_id: weight}`` map using the first pass to learn the
    canonical game order. Only resolvable post-``Benchmark.run`` because
    ``Game.game_id`` is only populated after ``start_game``.
    """
    first_pass = _pass_runs(benchmark, 0)
    if not first_pass:
        return {}
    weights = benchmark.game_weights if benchmark.game_weights is not None else [1.0] * len(first_pass)
    result: dict[str, float] = {}
    for i, run in enumerate(first_pass):
        if i < len(weights):
            result[run.game_id] = float(weights[i])
    return result


def _format_duration(td: timedelta) -> str:
    """Compact human-readable duration (h/m/s, integer seconds)."""
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_timestamp(dt: datetime) -> str:
    """A timestamp readable in a page header."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _timestamps_block_html(start: datetime | None, end: datetime | None, *, with_end: bool) -> str:
    """Inline HTML block for the start/end/duration block. ``with_end`` controls
    whether to show end-time + duration (used on the per-run page) or just
    start-time (comparison index, where each run is summarized briefly)."""
    if start is None:
        return ""
    parts: list[str] = [f"<p>Started: <code>{_format_timestamp(start)}</code></p>"]
    if with_end:
        if end is not None:
            parts.append(f"<p>Ended: <code>{_format_timestamp(end)}</code></p>")
            parts.append(f"<p>Duration: <code>{_format_duration(end - start)}</code></p>")
        else:
            parts.append("<p>Status: <em>in progress</em></p>")
    return "".join(parts)


def _git_status_block_html(benchmark: taaf.benchmark.Benchmark) -> str:
    """Embed ``job_dir/git_status.txt`` (written by the launcher) as a
    collapsible ``<details>`` block. Empty string if the file is
    missing — most often because ``job_dir`` is unset entirely.
    """
    if benchmark.job_dir is None:
        return ""
    path = benchmark.job_dir / "git_status.txt"
    if not path.exists():
        return ""
    return f"<details><summary>git status</summary><pre>{escape(path.read_text())}</pre></details>"


def _per_game_mean_scores(
    benchmark: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, float]:
    """Per-game mean score across passes. Weight 0 games are excluded
    (they should not appear in cross-game scatters / aggregates).
    ``score_fn`` overrides the per-run score (see ``_resolve_score_fn``)."""
    sf = _resolve_score_fn(scorer, score_fn)
    weights = weights_by_game_id(benchmark)
    out: dict[str, float] = {}
    for g in _game_ids(benchmark):
        if weights.get(g, 1.0) == 0.0:
            continue
        scores = [sf(r) for r in _game_runs_by_id(benchmark, g)]
        if scores:
            out[g] = statistics.mean(scores)
    return out


def _run_pass_stats(
    benchmark: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, float | None]:
    """Per-run mean / σ / SEM across passes.

    For each pass we take the weighted-mean per-game score (across all
    games in that pass, using ``Benchmark.game_weights``), then aggregate
    those per-pass means. ``sigma`` is the sample-stdev (ddof=1) of the
    per-pass means; ``sem`` = σ / √n_passes. Both are ``None`` when
    ``n_passes < 2`` — sample stdev is undefined for a single point.
    Weight-0 games are excluded from each pass's mean. ``score_fn``
    overrides the per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    weights = weights_by_game_id(benchmark)
    pass_means: list[float] = []
    for p in range(benchmark.n_passes):
        runs = _pass_runs(benchmark, p)
        kept = [(r, weights.get(r.game_id, 1.0)) for r in runs]
        kept = [(r, w) for r, w in kept if w > 0]
        if not kept:
            continue
        total_w = sum(w for _, w in kept)
        weighted_sum = sum(sf(r) * w for r, w in kept)
        pass_means.append(weighted_sum / total_w)
    if not pass_means:
        return {"mean": None, "sigma": None, "sem": None, "n_passes": 0}
    mean = float(np.mean(pass_means))
    if len(pass_means) < 2:
        return {"mean": mean, "sigma": None, "sem": None, "n_passes": len(pass_means)}
    sigma = float(np.std(pass_means, ddof=1))
    sem = sigma / (len(pass_means) ** 0.5)
    return {"mean": mean, "sigma": sigma, "sem": sem, "n_passes": len(pass_means)}


# --- Curve construction (plot semantics) ------------------------------------


def _record_results(run: taaf.game.GameRun) -> list[taaf.game.GameResult] | None:
    """Score every answer this run submitted, or ``None`` when they cannot be.

    From the recorded submissions and the run's own spec, for the report and
    never for the solver: ``Game.execute_action`` returns nothing precisely so
    a solver cannot read the correctness of an answer it can still replace,
    and computing it here keeps that true while still letting the curve show
    what a lane was worth as it spent.

    ``None`` when the spec was not recorded (``minimal_diagnostics``) or its
    answers were withheld (``simulate_submission``) — there is nothing to
    compare against, and a curve of zeros would read as a run that got
    everything wrong.
    """
    spec = run.spec
    if spec is None or not spec.answers_known:
        return None
    return [taaf.game.score_submission(spec, rec.submission) for rec in run.history]


def _per_game_partial_curve(
    run: taaf.game.GameRun,
    axis: CostAxis,
    *,
    scorer: Scorer = ARC_SCORER,
    input_weight: float = 0.0,
) -> list[tuple[float, float]]:
    """This lane's step events: ``(cost, score)`` at every point it changed.

    A solver may replace its answer, so a lane steps once per answer and the
    steps are **not monotonic** — replacing a right answer with a wrong one
    steps the lane back down, which is a real thing a run can do and the one
    thing this curve exists to make visible.

    Falls back to the two-point ``[(0, 0), (total, final)]`` shape when the
    intermediate answers cannot be scored (see :func:`_record_results`), which
    is also exactly what a lane with a single answer produces.

    ``input_weight`` charges uncached prompt tokens alongside generation --
    see :func:`charged_tokens`. It describes the token axis only, so asking
    for it on another axis raises rather than being quietly dropped.
    """
    if axis is TOKEN_AXIS:
        total_x = charged_tokens(run, input_weight)
    elif input_weight:
        raise ValueError("input_weight charges the generated-token axis; it has no meaning on another axis")
    else:
        total_x = float(axis.total(run))
    terminal = _score_for(run, scorer)
    results = _record_results(run)
    if not results:
        return [(0.0, 0.0), (total_x, terminal)]
    events = [(0.0, 0.0)]
    for i, result in enumerate(results):
        x = _charged_tokens_at_record(run, i, input_weight) if axis is TOKEN_AXIS else float(axis.at_record(run, i))
        y = scorer.of_result(result)
        # Several answers at the same cost (a solver reporting no tokens) are
        # one knot: the last one is what stood there.
        if events and abs(x - events[-1][0]) < 1e-9:
            events[-1] = (x, y)
        else:
            events.append((x, y))
    # The terminal event is authoritative — it carries the finish tail's cost
    # and the score the run actually reports.
    if abs(total_x - events[-1][0]) < 1e-9:
        events[-1] = (total_x, terminal)
    else:
        events.append((total_x, terminal))
    return events


def _pass_curve(
    pass_runs: list[taaf.game.GameRun],
    axis: CostAxis,
    *,
    scorer: Scorer = ARC_SCORER,
    weights_by_game_id: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean-cost-per-lane vs mean-score-per-lane step curve.

    Each input ``GameRun`` is one independent lane. All lanes run in
    parallel at equal speed; at parameter ``t`` lane ``i`` has spent
    ``min(t, T_i)`` cost. Both quantities are averaged (or weighted-
    averaged) across lanes. Knots are emitted wherever any lane steps.

    Decomposability: equal-weight pooling all ``(game, pass)`` runs in
    one call equals averaging passes within game then games. The
    function is agnostic to what the lanes mean — pass lanes within one
    pass, all-passes-of-one-game lanes, and everything-pooled lanes all
    use the same code path.

    Reading the curve: at ``x = X``, the value is the expected score
    when the method runs at an average per-game budget of ``X``.
    """
    if not pass_runs:
        return np.array([0.0]), np.array([0.0])
    lane_weights = (
        [weights_by_game_id.get(r.game_id, 1.0) for r in pass_runs] if weights_by_game_id is not None else None
    )
    return _pool_lane_events(
        [_per_game_partial_curve(r, axis, scorer=scorer) for r in pass_runs],
        weights=lane_weights,
    )


def _pool_lane_events(
    per_lane_events: list[list[tuple[float, float]]],
    weights: list[float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Step-curve combiner behind ``_pass_curve``.
    Returns the weighted-mean-cost vs weighted-mean-score
    step curve. ``weights`` defaults to equal-1.0. Lanes with weight 0
    are filtered out entirely (matches ``Benchmark.game_weights``'s
    "weight 0 = exclude from cross-game aggregation" semantics).

    Swept rather than re-read: at each knot only the lanes that actually
    stepped there change, and the cost accrues at a rate that is just the
    weight of the lanes still running. Re-deriving every lane's contribution
    at every knot would be quadratic in lanes times answers, which a solver
    that replaces its answer often makes expensive.

    Nothing here assumes the lanes rise. A lane that steps back down — a
    solver that replaced a right answer with a wrong one — pulls the mean
    down at that knot, which is the honest reading.
    """
    n = len(per_lane_events)
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != lanes length {n}")
    kept = [(evts, w) for evts, w in zip(per_lane_events, weights) if w > 0]
    if not kept:
        return np.array([0.0]), np.array([0.0])
    total_weight = sum(w for _, w in kept)

    # A lane's events are ascending in cost and its last event sits at its
    # total, so clamping to min(t, total) is the same as simply holding the
    # last value — no separate clamp is needed.
    steps: dict[float, list[tuple[int, float]]] = {}
    finishes: dict[float, float] = {}
    for lane, (evts, w) in enumerate(kept):
        for cx, cy in evts:
            steps.setdefault(cx, []).append((lane, cy))
        finishes[evts[-1][0]] = finishes.get(evts[-1][0], 0.0) + w

    current = [0.0] * len(kept)
    cost = 0.0
    score_sum = 0.0
    running_weight = total_weight
    previous = 0.0
    xs: list[float] = []
    ys: list[float] = []
    for t in sorted({0.0} | set(steps)):
        cost += (t - previous) * running_weight
        for lane, y in steps.get(t, ()):
            score_sum += kept[lane][1] * (y - current[lane])
            current[lane] = y
        # After the step: a lane finishing here still accrued cost up to t.
        running_weight -= finishes.get(t, 0.0)
        xs.append(cost / total_weight)
        ys.append(score_sum / total_weight)
        previous = t
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _job_clock_offset(run: taaf.game.GameRun, bench_start: datetime | None) -> float:
    """Seconds from the benchmark's start to this run's ``start_game``, so a
    game's own wallclock lands on the job clock. Zero when either stamp is
    missing (JSONs that predate ``GameRun.started_at``)."""
    if run.started_at is None or bench_start is None:
        return 0.0
    return (run.started_at - bench_start).total_seconds()


def _tokens_vs_wallclock_curve(
    benchmark: taaf.benchmark.Benchmark,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cumulative generated tokens across every action in every run as
    a function of job wallclock seconds (measured from
    ``benchmark.start_time``). Per-action x is
    ``(run.started_at - benchmark.start_time) + rec.wallclock_seconds``
    so games that queued behind a pool semaphore don't all collapse to
    benchmark t=0. Old JSONs that predate ``GameRun.started_at`` fall
    back to a zero offset (game-relative wallclock only).
    """
    events: list[tuple[float, int]] = []
    for run in benchmark.game_runs:
        offset = _job_clock_offset(run, benchmark.start_time)
        for rec in run.history:
            events.append((offset + float(rec.wallclock_seconds), int(rec.generated_tokens)))
        # No-move give-up / cancel tokens: charged at the finish
        # time (final_wallclock_seconds — same axis as per-action
        # wallclock_seconds), so they land after the last move.
        if run.final_generated_tokens:
            events.append((offset + float(run.final_wallclock_seconds), int(run.final_generated_tokens)))
    if not events:
        return np.array([0.0]), np.array([0.0])
    events.sort(key=lambda e: e[0])
    xs = np.empty(len(events) + 1, dtype=np.float64)
    ys = np.empty(len(events) + 1, dtype=np.float64)
    xs[0] = 0.0
    ys[0] = 0.0
    cum = 0
    for i, (t, n) in enumerate(events, start=1):
        cum += n
        xs[i] = t
        ys[i] = float(cum)
    return xs, ys


def _render_tokens_vs_wallclock_png(
    benchmarks: list[taaf.benchmark.Benchmark],
    title: str,
    *,
    labels: list[str] | None = None,
    palette: list[str] | None = None,
) -> str:
    """One curve per benchmark — single-curve on per-run pages, multi-
    curve on comparison pages (pass ``labels`` + ``RUN_COLORS``)."""
    return _render_curves_png(
        [_tokens_vs_wallclock_curve(b) for b in benchmarks],
        x_label="job wallclock (s)",
        y_label="cumulative generated tokens",
        title=title,
        labels=labels,
        palette=palette,
    )


def _score_vs_job_wallclock_curve(
    runs: list[taaf.game.GameRun],
    bench_start: datetime | None,
    end_seconds: float | None,
    *,
    scorer: Scorer = ARC_SCORER,
    weights_by_game_id: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mean score across ``runs`` as a function of job wallclock seconds.

    Unlike :func:`_pass_curve`, lanes are not slid onto a common origin: a
    lane scores 0 until its game started, and holds its last score once it
    ended, so a game queued behind a pool semaphore contributes from the time
    it actually ran. With equal weights, reading at ``x = T`` gives the mean
    score the job reported ``T`` seconds in. ``end_seconds`` (the job's length, or its
    elapsed time while in flight) extends the final plateau out to it.
    """
    if not runs:
        return np.array([0.0]), np.array([0.0])
    lanes: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for run in runs:
        offset = _job_clock_offset(run, bench_start)
        events = _per_game_partial_curve(run, WALLCLOCK_AXIS, scorer=scorer)
        lanes.append(
            (
                np.array([offset + x for x, _ in events], dtype=np.float64),
                np.array([y for _, y in events], dtype=np.float64),
            )
        )
    weights = (
        [weights_by_game_id.get(r.game_id, 1.0) for r in runs] if weights_by_game_id is not None else [1.0] * len(runs)
    )
    kept = [(lane, w) for lane, w in zip(lanes, weights) if w > 0]
    if not kept:
        return np.array([0.0]), np.array([0.0])
    knots = [0.0, *(x for (xs, _), _ in kept for x in xs)]
    if end_seconds is not None:
        knots.append(end_seconds)
    xs_out = np.unique(np.array(knots, dtype=np.float64))
    total_weight = sum(w for _, w in kept)
    ys_out = np.zeros_like(xs_out)
    for (lane_xs, lane_ys), w in kept:
        # Last event at or before each knot; ties at one x resolve to the later event.
        idx = np.searchsorted(lane_xs, xs_out, side="right") - 1
        ys_out += w * np.where(idx >= 0, lane_ys[np.maximum(idx, 0)], 0.0)
    return xs_out, ys_out / total_weight


def _benchmark_score_vs_job_wallclock_minutes(
    benchmark: taaf.benchmark.Benchmark,
    runs: list[taaf.game.GameRun] | None = None,
    *,
    scorer: Scorer,
    weights_by_game_id: dict[str, float] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """:func:`_score_vs_job_wallclock_curve` over a benchmark's runs (or a
    subset, e.g. one pass), with x converted to minutes for plotting."""
    xs, ys = _score_vs_job_wallclock_curve(
        benchmark.game_runs if runs is None else runs,
        benchmark.start_time,
        _benchmark_elapsed_seconds(benchmark),
        scorer=scorer,
        weights_by_game_id=weights_by_game_id,
    )
    return xs / 60.0, ys


def lane_score_at_budget(run: taaf.game.GameRun, budget: float, scorer: Scorer, *, input_weight: float = 0.0) -> float:
    """Score this single ``(game, pass)`` lane reached after spending
    ``budget`` generated tokens, read off its score-vs-tokens step
    curve. A lane that finished under budget keeps its terminal score.

    Public for analyses that compare runs at a spend they choose rather than
    at the one the runs happened to reach: a setting that wins more levels
    also earns more per-level allowance, so terminal scores compare
    configurations at different budgets. Note the terminal-score rule is only
    an extrapolation for a lane cut off from OUTSIDE (``state ==
    "cancelled"``); a lane that ended on its own would not have spent more.

    ``input_weight`` charges uncached prompt tokens into the same budget --
    see :func:`charged_tokens`. The default of zero leaves the budget
    generation-only, as it has always been.
    """
    events = _per_game_partial_curve(run, TOKEN_AXIS, scorer=scorer, input_weight=input_weight)
    total = events[-1][0]
    if budget >= total - 1e-9:
        return _score_for(run, scorer)
    cur = 0.0
    for cx, cy in events:
        if cx <= budget + 1e-9:
            cur = cy
        else:
            break
    return cur


def _capped_score_fn(scorer: Scorer, ceilings: dict[str, float]) -> Callable[[taaf.game.GameRun], float]:
    """Score callable that caps each lane at ITS OWN run's parallel-cutoff
    ceiling, keyed by ``run_id`` (see :func:`ceilings_by_run_id`).

    Keyed per lane rather than taking one ceiling, because the ceiling that
    realizes a budget depends on the spend distribution it is solved over. One
    ceiling shared by several runs makes their COMBINED mean equal the budget
    while leaving the heavier run reading at a higher per-game spend than the
    lighter one — the very difference a common budget exists to remove. Each
    run gets the ceiling that puts ITS OWN mean at the budget, which is also
    what reading that run's score-vs-tokens curve at the budget gives.
    """

    def _f(run: taaf.game.GameRun) -> float:
        return lane_score_at_budget(run, ceilings[run.run_id], scorer)

    return _f


def ceilings_by_run_id(benchmarks: list[taaf.benchmark.Benchmark], budget: float) -> dict[str, float]:
    """Each benchmark's own :func:`ceiling_for_budget`, mapped onto its lanes.

    ``run_id`` is globally unique, so one flat mapping addresses every lane of
    every run and the score callable reading it needs no benchmark context.
    """
    ceilings: dict[str, float] = {}
    for benchmark in benchmarks:
        ceiling = ceiling_for_budget([benchmark], budget)
        for run in benchmark.game_runs:
            ceilings[run.run_id] = ceiling
    return ceilings


def _min_tokens_per_game_budget(benchmarks: list[taaf.benchmark.Benchmark]) -> float:
    """Common per-game token budget for the capped statistics: the
    smallest across runs of each run's weighted-mean generated tokens
    per ``(game, pass)`` lane. This is the rightmost x at which every
    run's score-vs-tokens curve still has data — past it, the shortest
    run can't be read, so it's the fairest shared budget."""
    per_run: list[float] = []
    for b in benchmarks:
        weights = weights_by_game_id(b)
        kept = [(r, weights.get(r.game_id, 1.0)) for r in b.game_runs if weights.get(r.game_id, 1.0) > 0]
        if not kept:
            continue
        total_w = sum(w for _, w in kept)
        per_run.append(sum(total_tokens(r) * w for r, w in kept) / total_w)
    return min(per_run) if per_run else 0.0


def ceiling_for_budget(benchmarks: list[taaf.benchmark.Benchmark], budget: float) -> float:
    """Per-game token *ceiling* that realizes the common ``budget`` as a
    shared cutoff.

    Public alongside :func:`lane_score_at_budget`, and the reason the pair is
    needed rather than the lane function alone: a budget belongs to the
    ENSEMBLE, not to a game. One game overrunning is fine as long as the fleet
    averages to the budget, which is what this ceiling arranges. Mental model:
    every ``(game, pass)`` lane of the ensemble spends tokens in parallel, and
    we freeze the instant the weighted-mean spend reaches ``budget``. The
    freeze point is this ceiling — capping every lane at it makes the average
    lane spend exactly ``budget`` while leaving lanes that finished cheaper
    untouched and chopping only the expensive ones.

    **The ensemble is whatever is passed**, and comparing runs means passing
    one run at a time (:func:`ceilings_by_run_id`): the ceiling depends on the
    spend distribution it is solved over, so a ceiling solved over several
    runs at once equalizes their COMBINED mean and leaves the heavier run
    reading at a higher per-game spend than the lighter one.

    Capping directly at ``budget`` would undershoot: cheap lanes spend less
    than ``budget`` with nothing to offset them, so the average falls below
    it and heterogeneous runs get understated. Returns ``inf`` when
    ``budget`` already meets or exceeds the grand mean (nothing to cap).
    """
    lanes: list[tuple[float, float]] = []
    for b in benchmarks:
        weights = weights_by_game_id(b)
        for r in b.game_runs:
            w = weights.get(r.game_id, 1.0)
            if w > 0:
                lanes.append((float(total_tokens(r)), w))
    return ceiling_for_lanes(lanes, budget)


def ceiling_for_lanes(lanes: list[tuple[float, float]], budget: float) -> float:
    """The solve under :func:`ceiling_for_budget`, over explicit ``(spend,
    weight)`` lanes: the ``C`` with ``sum(w * min(C, T)) == budget * sum(w)``.

    Public for callers whose ensemble is not a whole benchmark — a sweep
    scoring each ARM at its own budget hands one arm's lanes at a time, and
    duplicating the solve is how the two quietly disagree. Zero-weight lanes
    are the caller's to exclude; ``inf`` when the weighted mean spend is
    already at or under ``budget`` (the lanes ended on their own, and more
    budget would not have changed them).
    """
    if not lanes:
        return 0.0
    total_w = sum(w for _, w in lanes)
    target_sum = budget * total_w  # solve Σ w·min(C, T) == target_sum for C
    if target_sum >= sum(t * w for t, w in lanes) - 1e-9:
        return float("inf")
    lanes.sort()
    sum_fixed = 0.0  # Σ w·T over lanes already below the ceiling
    w_tail = total_w  # weight of lanes still at or above the ceiling
    for t, w in lanes:
        # For C in [prev t, this t], Σ w·min(C, T) = sum_fixed + C·w_tail,
        # which peaks at C = t; the target lands in the first such interval.
        if sum_fixed + t * w_tail >= target_sum:
            return (target_sum - sum_fixed) / w_tail
        sum_fixed += w * t
        w_tail -= w
    return float("inf")  # unreachable given the grand-mean guard above


# --- PNG rendering ----------------------------------------------------------


def _fig_to_png_b64(fig: matplotlib.figure.Figure) -> str:
    """Render a figure to base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


#: Cell value -> RGB, as one lookup table rather than a colormap call per pixel.
_ARC_RGB: npt.NDArray[np.uint8] = np.array(
    [[int(code[1:3], 16), int(code[3:5], 16), int(code[5:7], 16)] for code in taaf.grid.ARC_COLORS],
    dtype=np.uint8,
)


def _grid_png_b64(grid: taaf.grid.Grid) -> str:
    """Native-resolution PNG of one grid: exactly one pixel per cell, so
    CSS can zoom it with ``image-rendering: pixelated`` and get whole
    cells.

    Written from the value array directly rather than through a figure.
    A figure puts the grid inside an axes, and an axes is placed in device
    space by a renderer that rounds and that enforces its own aspect
    ratio; at one pixel per cell those sub-pixel corrections cost the
    outer row and column, which fall off the canvas entirely.
    """
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, _ARC_RGB[grid.data], format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grid_img_html(grid: taaf.grid.Grid, width_px: int, alt: str = "") -> str:
    """One grid as an inline pixel-art image."""
    return (
        f'<img class="pixelart" width="{width_px}" '
        f'src="data:image/png;base64,{_grid_png_b64(grid)}" alt="{escape(alt)}">'
    )


def _thumbnail_html(run: taaf.game.GameRun | None, width_px: int = THUMB_DEFAULT_PX) -> str:
    """A task's mini-figure: the first question it has to answer.

    One image identifies a task at a glance, and the question is the only
    part present in every mode — a run with withheld answers has none to
    show.
    """
    if run is None or run.spec is None:
        return ""
    return _grid_img_html(run.spec.to_answer[0].question, width_px, "first question")


def _render_curves_png(
    curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    labels: list[str] | None = None,
    palette: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
    overlay_curves: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None,
) -> str:
    """Multi-curve step plot. ``palette`` selects the color cycle
    (default ``ACCENT_COLORS``; comparison plots use ``RUN_COLORS``).
    ``overlay_curves`` are drawn faint under the main curves — used to
    show per-pass variance under a pooled curve without losing the
    pooled curve's budget interpretation. No legend entry for overlays.
    """
    pal = palette if palette is not None else ACCENT_COLORS
    fig = plt.figure(figsize=figsize, dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    if overlay_curves:
        for xs, ys in overlay_curves:
            ax.step(xs, ys, where="post", color=pal[0], alpha=0.45, linewidth=0.7, zorder=1)
    for i, (xs, ys) in enumerate(curves):
        label = labels[i] if labels else None
        ax.step(xs, ys, where="post", color=pal[i % len(pal)], label=label, linewidth=1.8, zorder=3)
    _style_dark(ax, x_label, y_label, title)
    return _fig_to_png_b64(fig)


def _draw_labelled_points(
    ax: matplotlib.axes.Axes,
    xs: list[float],
    ys: list[float],
    labels: list[str],
    color: str = ACCENT_COLORS[0],
) -> None:
    """Scatter labelled points in the dark-theme style. Shared by per-game
    score-vs-tokens (per-run) and head-to-head per-game (comparison)."""
    ax.scatter(xs, ys, color=color, s=70, edgecolors=TEXT_COLOR, linewidths=0.5, zorder=3)
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", color=TEXT_COLOR, fontsize=8)


def _render_scatter_png(
    per_game_a: dict[str, float],
    per_game_b: dict[str, float],
    a_label: str,
    b_label: str,
    title: str,
) -> str:
    """Per-game score scatter of one run against another, with the diagonal
    drawn so a point's side says which run won that game."""
    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    games = sorted(set(per_game_a) & set(per_game_b))
    xs = [per_game_a[g] for g in games]
    ys = [per_game_b[g] for g in games]
    _draw_labelled_points(ax, xs, ys, games)
    bound = max(max(xs, default=0.0), max(ys, default=0.0), 1.0) * 1.05
    ax.plot([0, bound], [0, bound], color=GRID_COLOR, linestyle="--", linewidth=1, label="y = x", zorder=1)
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_aspect("equal")
    _style_dark(ax, a_label, b_label, title)
    return _fig_to_png_b64(fig)


def _render_per_game_score_vs_tokens_scatter_png(benchmark: taaf.benchmark.Benchmark, title: str) -> str:
    """One labelled point per game: x = mean tokens across passes, y = mean
    score across passes. Complements the pooled-lane curve by exposing which
    games are cheap-and-easy vs. expensive-and-hard."""
    fig = plt.figure(figsize=(7, 5), dpi=100, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    games = _game_ids(benchmark)
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    for g in games:
        runs = _game_runs_by_id(benchmark, g)
        if not runs:
            continue
        xs.append(float(statistics.mean(total_tokens(r) for r in runs)))
        ys.append(float(statistics.mean(_arc_score(r) for r in runs)))
        labels.append(g)
    _draw_labelled_points(ax, xs, ys, labels)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    _style_dark(ax, "mean tokens per pass", "mean score per pass", title)
    return _fig_to_png_b64(fig)


def _style_dark(ax: matplotlib.axes.Axes, x_label: str, y_label: str, title: str) -> None:
    """Apply the page palette to an axes."""
    ax.set_xlabel(x_label, color=TEXT_COLOR)
    ax.set_ylabel(y_label, color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Anchor legend outside the axes to avoid overlapping curves;
        # bbox_inches="tight" extends the figure bbox to include it.
        legend = ax.legend(
            handles,
            labels,
            facecolor=BG_COLOR,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )
        legend.get_frame().set_alpha(0.9)


# --- Statistical test -------------------------------------------------------


def _pass_level_score_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Welch's (unpaired) t-test on per-pass mean scores. Each pass is
    one observation (mean per-game score in that pass).

    Pass-level rather than per-game pairing because solvers tend to
    excel on different games — there's no shared game ranking, so
    per-game pairing wouldn't cancel noise.

    Returns NaN ``t`` / ``p`` / ``df`` when either side has < 2
    passes. At n = 2 per side Welch df ≈ 1 — essentially powerless;
    bump n_passes to 3+ for a usable p. ``score_fn`` overrides the
    per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    weights_a = weights_by_game_id(bm_a)
    weights_b = weights_by_game_id(bm_b)
    pass_means_a: list[float] = []
    pass_means_b: list[float] = []
    for p in range(bm_a.n_passes):
        kept = [(r, weights_a.get(r.game_id, 1.0)) for r in _pass_runs(bm_a, p) if weights_a.get(r.game_id, 1.0) > 0]
        if kept:
            total_w = sum(w for _, w in kept)
            pass_means_a.append(sum(sf(r) * w for r, w in kept) / total_w)
    for p in range(bm_b.n_passes):
        kept = [(r, weights_b.get(r.game_id, 1.0)) for r in _pass_runs(bm_b, p) if weights_b.get(r.game_id, 1.0) > 0]
        if kept:
            total_w = sum(w for _, w in kept)
            pass_means_b.append(sum(sf(r) * w for r, w in kept) / total_w)
    overall_a = float(np.mean(pass_means_a)) if pass_means_a else 0.0
    overall_b = float(np.mean(pass_means_b)) if pass_means_b else 0.0
    if len(pass_means_a) < 2 or len(pass_means_b) < 2:
        return {
            "mean_a": overall_a,
            "mean_b": overall_b,
            "mean_diff": overall_b - overall_a,
            "t": float("nan"),
            "p": float("nan"),
            "df": float("nan"),
            "n_passes_a": len(pass_means_a),
            "n_passes_b": len(pass_means_b),
            "zero_variance": False,
        }
    # Zero sample variance on either side: Welch's can't estimate that
    # side's noise. scipy returns t = ±inf / p ∈ {0, NaN} which reads
    # like a real significance estimate. Flag so the renderer can say
    # "not meaningful" instead of showing the bogus number.
    zero_variance = float(np.var(pass_means_a, ddof=1)) == 0.0 or float(np.var(pass_means_b, ddof=1)) == 0.0
    res: Any = scipy.stats.ttest_ind(pass_means_b, pass_means_a, equal_var=False)
    return {
        "mean_a": overall_a,
        "mean_b": overall_b,
        "mean_diff": overall_b - overall_a,
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "df": float(res.df) if hasattr(res, "df") else float("nan"),
        "n_passes_a": len(pass_means_a),
        "n_passes_b": len(pass_means_b),
        "zero_variance": zero_variance,
    }


def _weighted_paired_t_test(diffs: list[float], weights: list[float]) -> dict[str, Any]:
    """Weighted paired t-test on ``diffs`` with parallel per-pair
    ``weights``. Weights are treated as importance / reliability, not
    frequencies — each pair is one independent observation; weights
    rescale contribution but don't multiply sample size. Using
    ``sum(w) - 1`` as df would overstate certainty and produce
    p-values that disagree with the paired permutation test.

    Formulae::

        mean_w = sum(w·d) / sum(w)
        var_w  = sum(w·(d − mean_w)²) / (sum(w) − sum(w²)/sum(w))
                                          # analytic-weight Bessel
                                          # analogue, scale-invariant
                                          # under uniform rescaling
        n_eff  = (sum w)² / sum(w²)       # Kish effective N
        SE     = √(var_w / n_eff)
        t      = mean_w / SE
        df     = n_eff − 1
        p      = 2·P(T > |t|),  T ~ Student-t(df)

    With all weights 1.0, ``n_eff = n`` and this reduces to
    ``scipy.stats.ttest_rel`` (asserted by
    ``test_weighted_paired_t_matches_scipy_when_uniform``).
    """
    if len(diffs) != len(weights):
        raise ValueError(f"diffs length {len(diffs)} != weights length {len(weights)}")
    n = len(diffs)
    out_nan = {
        "t": float("nan"),
        "p": float("nan"),
        "df": float("nan"),
        "n_pairs": n,
        "zero_variance": False,
    }
    if n < 2:
        return out_nan
    d = np.asarray(diffs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(w))
    sw2 = float(np.sum(w * w))
    if sw <= 0.0 or sw2 <= 0.0:
        return out_nan
    n_eff = sw * sw / sw2
    if n_eff <= 1.0:  # df = n_eff - 1 ≤ 0 — undefined
        return out_nan
    mean_w = float(np.sum(w * d) / sw)
    # ``sw - sw2/sw = (sw² - sw2)/sw`` equals n-1 when w_i = 1 and is
    # invariant under uniform rescaling (doubling every weight leaves t).
    denom = sw - sw2 / sw
    if denom <= 0.0:  # only one nonzero weight — undefined
        return out_nan
    var_w = float(np.sum(w * (d - mean_w) ** 2) / denom)
    if var_w == 0.0:
        return {**out_nan, "zero_variance": True}
    se = (var_w / n_eff) ** 0.5
    t = mean_w / se
    df = n_eff - 1.0
    p = 2.0 * float(scipy.stats.t.sf(abs(t), df))  # pyright: ignore[reportUnknownArgumentType]
    return {"t": t, "p": p, "df": df, "n_pairs": n, "zero_variance": False}


# 2^20 ≈ 1M sign-flips runs in a vectorised numpy step in well under
# a second; beyond that we switch to Monte Carlo.
_PERMUTATION_EXACT_MAX_N = 20
_PERMUTATION_MC_SAMPLES = 10000


def _weighted_paired_permutation_test(diffs: list[float], weights: list[float]) -> dict[str, Any]:
    """Weighted paired sign-flip permutation test on ``diffs`` with
    parallel per-pair ``weights``. Statistic
    ``T = sum(w·d) / sum(w)``; under the exchangeability null each
    sign is equally likely. Exact enumeration of all 2^n sign-flips
    for ``n ≤ _PERMUTATION_EXACT_MAX_N``, else Monte Carlo with the
    ``(count+1)/(N+1)`` correction. Two-sided
    ``p = P(|T_permuted| ≥ |T_observed|)``.

    Reduces to the standard sign-flip test when weights are uniform
    (asserted by ``test_weighted_permutation_matches_explicit_when_uniform``).
    """
    if len(diffs) != len(weights):
        raise ValueError(f"diffs length {len(diffs)} != weights length {len(weights)}")
    n = len(diffs)
    if n < 2:
        return {
            "p": float("nan"),
            "n_pairs": n,
            "exact": False,
            "n_permutations": 0,
            "zero_variance": False,
        }
    d = np.asarray(diffs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return {
            "p": float("nan"),
            "n_pairs": n,
            "exact": False,
            "n_permutations": 0,
            "zero_variance": False,
        }
    zero_variance = bool(np.all(d == 0.0))
    observed = float(np.sum(w * d) / sw)
    abs_obs = abs(observed)
    if zero_variance:
        return {
            "p": 1.0,
            "n_pairs": n,
            "exact": True,
            "n_permutations": 1 << n,
            "zero_variance": True,
        }
    if n <= _PERMUTATION_EXACT_MAX_N:
        masks = np.arange(1 << n, dtype=np.int64)[:, None]
        bits = (masks >> np.arange(n)) & 1
        signs = 1 - 2 * bits  # (2^n, n)
        permuted_means = (signs * w * d).sum(axis=1) / sw
        count_as_extreme = int(np.sum(np.abs(permuted_means) >= abs_obs - 1e-12))
        n_perms = 1 << n
        p = count_as_extreme / n_perms
        return {
            "p": p,
            "n_pairs": n,
            "exact": True,
            "n_permutations": n_perms,
            "zero_variance": False,
        }
    # Fixed seed derived from the observed statistic so the reported p
    # is reproducible across regens, with distinct seeds across datasets.
    rng = np.random.default_rng(abs(hash(observed)) & 0xFFFFFFFF)
    signs = rng.choice(np.array([-1, 1], dtype=np.float64), size=(_PERMUTATION_MC_SAMPLES, n))
    permuted_means = (signs * w * d).sum(axis=1) / sw
    count_as_extreme = int(np.sum(np.abs(permuted_means) >= abs_obs - 1e-12))
    p = (count_as_extreme + 1) / (_PERMUTATION_MC_SAMPLES + 1)
    return {
        "p": p,
        "n_pairs": n,
        "exact": False,
        "n_permutations": _PERMUTATION_MC_SAMPLES,
        "zero_variance": False,
    }


def _paired_score_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Weighted paired t-test on per-game mean scores. One pair per
    common game (mean across passes under each solver), per-pair weight
    averaged from the two benchmarks' ``game_weights``. Weight-0 games
    are excluded. Same null hypothesis as ``_pass_level_score_test``;
    per-game pairing wins when game difficulty correlates across
    solvers, pass-level wins otherwise. ``score_fn`` overrides the
    per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    common = sorted(set(_game_ids(bm_a)) & set(_game_ids(bm_b)))
    weights_a = weights_by_game_id(bm_a)
    weights_b = weights_by_game_id(bm_b)
    diffs: list[float] = []
    pair_weights: list[float] = []
    for g in common:
        wa = weights_a.get(g, 1.0)
        wb = weights_b.get(g, 1.0)
        if wa == 0.0 or wb == 0.0:
            continue
        a_scores = [sf(r) for r in _game_runs_by_id(bm_a, g)]
        b_scores = [sf(r) for r in _game_runs_by_id(bm_b, g)]
        if not a_scores or not b_scores:
            continue
        diffs.append(statistics.mean(b_scores) - statistics.mean(a_scores))
        # Per-pair weight averages the two sides; identical when both
        # benchmarks carry the same game_weights (the common case).
        pair_weights.append((wa + wb) / 2.0)
    result = _weighted_paired_t_test(diffs, pair_weights)
    return {
        "t": result["t"],
        "p": result["p"],
        "df": result["df"],
        "n_games": result["n_pairs"],
        "zero_variance": result["zero_variance"],
    }


def _paired_permutation_test(
    bm_a: taaf.benchmark.Benchmark,
    bm_b: taaf.benchmark.Benchmark,
    *,
    scorer: Scorer = ARC_SCORER,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> dict[str, Any]:
    """Weighted paired sign-flip permutation test on per-game mean
    differences. Same per-game pairing as ``_paired_score_test``;
    robust to non-Gaussian / heavy-tailed / skewed differences where
    the paired t-test's normality assumption is suspect. ``score_fn``
    overrides the per-run score (see ``_resolve_score_fn``).
    """
    sf = _resolve_score_fn(scorer, score_fn)
    common = sorted(set(_game_ids(bm_a)) & set(_game_ids(bm_b)))
    weights_a = weights_by_game_id(bm_a)
    weights_b = weights_by_game_id(bm_b)
    diffs: list[float] = []
    pair_weights: list[float] = []
    for g in common:
        wa = weights_a.get(g, 1.0)
        wb = weights_b.get(g, 1.0)
        if wa == 0.0 or wb == 0.0:
            continue
        a_scores = [sf(r) for r in _game_runs_by_id(bm_a, g)]
        b_scores = [sf(r) for r in _game_runs_by_id(bm_b, g)]
        if not a_scores or not b_scores:
            continue
        diffs.append(statistics.mean(b_scores) - statistics.mean(a_scores))
        pair_weights.append((wa + wb) / 2.0)
    result = _weighted_paired_permutation_test(diffs, pair_weights)
    return {
        "p": result["p"],
        "n_games": result["n_pairs"],
        "exact": result["exact"],
        "n_permutations": result["n_permutations"],
        "zero_variance": result["zero_variance"],
    }


def _welch_ok(res: dict[str, Any]) -> bool:
    """Welch's t-test is reportable only with ≥ 2 passes on each side."""
    return res["n_passes_a"] >= 2 and res["n_passes_b"] >= 2


def _paired_ok(res: dict[str, Any]) -> bool:
    """The paired tests need ≥ 2 common games to have any signal."""
    return res["n_games"] >= 2


def _direction_arrow(
    mean_row: float | None,
    mean_col: float | None,
    row_label: str,
    col_label: str,
) -> tuple[str, str]:
    """``(arrow, winner_label)`` pointing toward the higher-scoring run:
    ``←`` when this cell's row run scores higher, ``↑`` when its column
    run does. Empty when tied or either mean is unknown."""
    if mean_row is None or mean_col is None or mean_row == mean_col:
        return "", ""
    return ("←", row_label) if mean_row > mean_col else ("↑", col_label)


def _pvalue_cell_html(res: dict[str, Any], ok: bool, *, arrow: str = "", winner: str = "") -> str:
    """One ``<td>`` for a pairwise p-value matrix: ``n/a`` when the test
    is unreportable (too few passes/games), ``n.m.`` when a zero-variance
    side makes the p meaningless, else the two-sided p (bold if < 0.05).
    An optional direction ``arrow`` (← row run higher, ↑ column run
    higher) points toward the better-scoring run, ``winner`` labels it."""
    prefix = f'<span class="pval-arrow" title="{escape(winner)} scores higher">{arrow}</span> ' if arrow else ""
    if not ok:
        return f'<td class="pval-na" title="too few passes / games for this test">{prefix}n/a</td>'
    if res.get("zero_variance"):
        return f'<td class="pval-na" title="zero sample variance — p not meaningful">{prefix}n.m.</td>'
    p = res["p"]
    if np.isnan(p):
        return f'<td class="pval-na" title="undefined">{prefix}n/a</td>'
    cls = "pval-sig" if p < 0.05 else "pval"
    return f'<td class="{cls}">{prefix}{p:.4f}</td>'


def _render_pvalue_matrix(
    title: str,
    labels: list[str],
    results: dict[tuple[int, int], dict[str, Any]],
    ok_fn: Callable[[dict[str, Any]], bool],
    means: list[float | None],
) -> str:
    """One symmetric NxN p-value matrix. ``results`` is keyed by the
    upper-triangle pair ``(i, j)`` with ``i < j``; the two-sided p is
    symmetric so cell ``(j, i)`` reuses it and the diagonal is blank.
    ``means`` (per-run mean score, parallel to ``labels``) drives the
    per-cell direction arrow toward the higher-scoring run."""
    header_cells = "".join(f"<th><code>{escape(lbl)}</code></th>" for lbl in labels)
    rows = [f"<tr><th></th>{header_cells}</tr>"]
    for i, li in enumerate(labels):
        cells = [f"<th><code>{escape(li)}</code></th>"]
        for j in range(len(labels)):
            if i == j:
                cells.append('<td class="pval-diag">—</td>')
                continue
            res = results[(i, j) if i < j else (j, i)]
            arrow, winner = _direction_arrow(means[i], means[j], li, labels[j])
            cells.append(_pvalue_cell_html(res, ok_fn(res), arrow=arrow, winner=winner))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<h4>{title}</h4><table class="pmatrix">{"".join(rows)}</table>'


def _pvalue_matrices_inner(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """With more than two runs: the same three tests as the 2-run head-to-head,
    each rendered as a symmetric NxN matrix of pairwise two-sided
    p-values. Returns the legend + matrices; the caller wraps it in a
    box. ``score_fn`` selects endpoint vs common-budget scoring."""
    n = len(benchmarks)
    # Tests are two-sided ⇒ p is symmetric in the pair; compute the
    # upper triangle once and mirror it in the renderer.
    welch: dict[tuple[int, int], dict[str, Any]] = {}
    paired: dict[tuple[int, int], dict[str, Any]] = {}
    perm: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            welch[(i, j)] = _pass_level_score_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
            paired[(i, j)] = _paired_score_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
            perm[(i, j)] = _paired_permutation_test(benchmarks[i], benchmarks[j], scorer=scorer, score_fn=score_fn)
    # Per-run mean score (same scoring as the cells) drives the direction
    # arrow toward the higher-scoring run in each cell.
    means = [_run_pass_stats(b, scorer=scorer, score_fn=score_fn)["mean"] for b in benchmarks]
    matrices = (
        _render_pvalue_matrix("Welch's t-test — per-pass mean scores", labels, welch, _welch_ok, means)
        + _render_pvalue_matrix("Paired t-test — per-game mean scores", labels, paired, _paired_ok, means)
        + _render_pvalue_matrix("Paired permutation test — per-game differences", labels, perm, _paired_ok, means)
    )
    return (
        "<p>Each test asks whether two runs differ in mean per-game score — same null "
        "hypothesis, different variance / null structure (see the 2-run page for the full "
        "explanation). Cells are two-sided p-values; each matrix is symmetric and the "
        "diagonal is blank. The arrow points to the higher-scoring run "
        '(<span class="pval-arrow">←</span> this row, <span class="pval-arrow">↑</span> this '
        "column). <strong>Bold</strong> marks p &lt; 0.05. "
        "<code>n.m.</code> = not meaningful (zero variance on a side); "
        "<code>n/a</code> = too few passes / games.</p>"
        f"{matrices}"
    )


def _fmt_stat(v: float | None) -> str:
    """A statistic to two places, or an em dash when it does not exist."""
    return "—" if v is None else f"{v:.2f}"


def _two_run_stats_inner(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """Head-to-head inner content for exactly two runs: per-run mean
    table (winner bolded with 🏆), the winner line, and the three
    p-value test lines. Returns inner HTML; the caller wraps it in a
    box. ``score_fn`` selects endpoint vs common-budget scoring."""
    result = _pass_level_score_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    stats_a = _run_pass_stats(benchmarks[0], scorer=scorer, score_fn=score_fn)
    stats_b = _run_pass_stats(benchmarks[1], scorer=scorer, score_fn=score_fn)
    a_label, b_label = labels[0], labels[1]
    a_mean, b_mean = result["mean_a"], result["mean_b"]

    # Highlight the winner with bold + 🏆; tie if exactly equal.
    if a_mean > b_mean:
        winner_text = f"<strong>{escape(a_label)}</strong> ahead by {a_mean - b_mean:.2f} points"
        row_a_score = f"<strong>{a_mean:.2f}</strong> 🏆"
        row_b_score = f"{b_mean:.2f}"
    elif b_mean > a_mean:
        winner_text = f"<strong>{escape(b_label)}</strong> ahead by {b_mean - a_mean:.2f} points"
        row_a_score = f"{a_mean:.2f}"
        row_b_score = f"<strong>{b_mean:.2f}</strong> 🏆"
    else:
        winner_text = "Tied."
        row_a_score = f"{a_mean:.2f}"
        row_b_score = f"{b_mean:.2f}"

    # Welch's needs n ≥ 2 per side; n = 2 gives df ≈ 1 and is
    # effectively powerless. Zero variance on either side makes p
    # meaningless — we report that distinctly rather than letting
    # p = 0.0000 look real.
    n_a, n_b = result["n_passes_a"], result["n_passes_b"]
    if n_a >= 2 and n_b >= 2:
        if result.get("zero_variance"):
            welch_line = (
                f"<li><strong>Welch's t-test on per-pass mean scores</strong> "
                f"(n_passes = {n_a} vs {n_b}): "
                f"<strong>not meaningful</strong> — at least one side has zero "
                f"sample variance across passes (deterministic solver, or all "
                f"passes produced identical mean scores).</li>"
            )
        else:
            welch_caveat = (
                " <em>With only 2 passes per side, df ≈ 1 — essentially powerless. "
                "Re-run with more passes for a usable p.</em>"
                if n_a == 2 or n_b == 2
                else ""
            )
            welch_line = (
                f"<li><strong>Welch's t-test on per-pass mean scores</strong> "
                f"(each pass treated as one independent observation; "
                f"n_passes = {n_a} vs {n_b}, df = {result['df']:.2f}, t = {result['t']:.3f}): "
                f"<strong>p = {result['p']:.4f}</strong>.{welch_caveat}</li>"
            )
    else:
        welch_line = (
            f"<li><strong>Welch's t-test on per-pass mean scores</strong>: "
            f"too few passes ({n_a} vs {n_b}) — need ≥ 2 per side.</li>"
        )

    # Same null hypothesis as Welch's but per-game pairing — wins when
    # the two solvers correlate across games (similar relative difficulty).
    paired = _paired_score_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    if paired["n_games"] >= 2:
        if paired.get("zero_variance"):
            paired_line = (
                f"<li><strong>Paired t-test on per-game mean scores</strong> "
                f"(n_games = {paired['n_games']}): "
                f"<strong>not meaningful</strong> — the per-game differences "
                f"have zero variance across games (identical mean scores per "
                f"game between the two solvers, so the paired test has no "
                f"signal to estimate).</li>"
            )
        else:
            paired_line = (
                f"<li><strong>Paired t-test on per-game mean scores</strong> "
                f"(pairs each game's mean under A with its mean under B; "
                f"n_games = {paired['n_games']}, df = {paired['df']:.0f}, t = {paired['t']:.3f}): "
                f"<strong>p = {paired['p']:.4f}</strong>.</li>"
            )
    else:
        paired_line = (
            f"<li><strong>Paired t-test on per-game mean scores</strong>: "
            f"too few games ({paired['n_games']}) — need ≥ 2.</li>"
        )

    # Robust to non-Gaussian / heavy-tailed per-game differences where
    # the paired t-test's normality assumption is suspect.
    permutation = _paired_permutation_test(benchmarks[0], benchmarks[1], scorer=scorer, score_fn=score_fn)
    if permutation["n_games"] >= 2:
        if permutation.get("zero_variance"):
            permutation_line = (
                f"<li><strong>Paired permutation test on per-game mean scores</strong> "
                f"(n_games = {permutation['n_games']}): "
                f"<strong>not meaningful</strong> — every per-game difference is zero, "
                f"so every sign-flip yields the same statistic as the observed.</li>"
            )
        else:
            method = (
                f"exact, {permutation['n_permutations']} sign-flips"
                if permutation["exact"]
                else f"Monte Carlo, {permutation['n_permutations']} samples"
            )
            permutation_line = (
                f"<li><strong>Paired permutation test on per-game mean scores</strong> "
                f"(sign-flip null on per-game differences; {method}; "
                f"n_games = {permutation['n_games']}): "
                f"<strong>p = {permutation['p']:.4f}</strong>.</li>"
            )
    else:
        permutation_line = (
            f"<li><strong>Paired permutation test on per-game mean scores</strong>: "
            f"too few games ({permutation['n_games']}) — need ≥ 2.</li>"
        )

    test_line = (
        "<p>Three tests below ask the same question — does the mean per-game score differ "
        "between the two runs? They differ in the variance / null structure they assume:</p>"
        f"<ul>{welch_line}{paired_line}{permutation_line}</ul>"
    )
    sigma_caveat = (
        ""
        if min(stats_a["n_passes"] or 0, stats_b["n_passes"] or 0) >= 2
        else "<p><em>σ across passes is undefined for runs with only one pass.</em></p>"
    )
    return (
        "<table>"
        "<tr><th>run</th><th>mean per-game score</th>"
        "<th>σ across passes</th><th>σ/√N (SEM)</th></tr>"
        f"<tr><td><code>{escape(a_label)}</code></td><td>{row_a_score}</td>"
        f"<td>{_fmt_stat(stats_a['sigma'])}</td><td>{_fmt_stat(stats_a['sem'])}</td></tr>"
        f"<tr><td><code>{escape(b_label)}</code></td><td>{row_b_score}</td>"
        f"<td>{_fmt_stat(stats_b['sigma'])}</td><td>{_fmt_stat(stats_b['sem'])}</td></tr>"
        "</table>"
        f"<p>{winner_text}.</p>"
        f"{sigma_caveat}"
        f"{test_line}"
    )


def _comparison_stats_box(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None,
    heading: str,
    note: str,
) -> str:
    """One score-comparison summary box for a given per-run score
    function. Two runs → head-to-head; more than two → pairwise p-value
    matrices. ``heading`` / ``note`` distinguish the endpoint and
    common-budget variants rendered side by side per scorer."""
    if len(benchmarks) == 2:
        inner = _two_run_stats_inner(benchmarks, labels, scorer=scorer, score_fn=score_fn)
    else:
        inner = _pvalue_matrices_inner(benchmarks, labels, scorer=scorer, score_fn=score_fn)
    note_html = f"<p>{note}</p>" if note else ""
    return f'<div class="summary-box"><h3>{escape(heading)}</h3>{note_html}{inner}</div>'


def _per_run_stats_box(
    benchmarks: list[taaf.benchmark.Benchmark],
    *,
    scorer: Scorer,
    score_fn: Callable[[taaf.game.GameRun], float] | None = None,
) -> str:
    """At-a-glance per-run mean / σ across passes / SEM box (all runs).
    ``score_fn`` selects endpoint vs common-budget scoring."""
    rows = ["<tr><th>run</th><th>passes</th><th>mean</th><th>σ across passes</th><th>SEM</th></tr>"]
    for b in benchmarks:
        s = _run_pass_stats(b, scorer=scorer, score_fn=score_fn)
        rows.append(
            f"<tr><td><code>{escape(b.label)}</code></td>"
            f"<td>{s['n_passes']}</td>"
            f"<td>{_fmt_stat(s['mean'])}</td>"
            f"<td>{_fmt_stat(s['sigma'])}</td>"
            f"<td>{_fmt_stat(s['sem'])}</td></tr>"
        )
    return '<div class="summary-box"><h3>Per-run stats</h3><table>' + "".join(rows) + "</table></div>"


# --- HTML helpers -----------------------------------------------------------


_CSS = """
body { background: #1e1e1e; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
       padding: 20px; max-width: 1400px; margin: 0 auto; line-height: 1.4; }
h1, h2, h3 { color: #ffffff; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { border: 1px solid #3a3a3a; padding: 6px 10px; text-align: left; vertical-align: middle; }
th { background: #2a2a2a; }
.pixelart { image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges;
            vertical-align: middle; }
a { color: #9bd1ff; }
a:visited { color: #c9b1ff; }
pre { background: #2a2a2a; padding: 10px; border-radius: 4px; overflow-x: auto; }
.game-name { display: inline-flex; align-items: center; gap: 6px; }
.summary-box { background: #2a2a2a; padding: 12px 16px; border-radius: 4px; margin: 12px 0; }
.game-link { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px;
             margin: 4px; background: #2a2a2a; border-radius: 4px; text-decoration: none; }
.intro { color: #b8b8b8; font-style: italic; margin: 4px 0 16px 0; max-width: 70ch; }
.design-note { background: #3a2a1e; border-left: 4px solid #ffb47a; padding: 10px 14px;
               margin: 12px 0; border-radius: 4px; }
.curve-note { background: #1e2e2a; border-left: 4px solid #74a98c; padding: 10px 14px;
              margin: 12px 0; border-radius: 4px; max-width: 80ch; }
.curve-note p { margin: 6px 0; }
.curve-note code { background: #2a3a36; padding: 1px 4px; border-radius: 2px; }
.in-flight-banner { background: #1e2e3a; border-left: 4px solid #5a92c8; padding: 12px 16px;
                    margin: 12px 0; border-radius: 4px; font-size: 1.05em; color: #cfe2f3; }
.weights-banner { background: #2a2a1e; border-left: 4px solid #f0d878; padding: 12px 16px;
                  margin: 12px 0; border-radius: 4px; color: #f4e8c0; }
.weights-banner p { margin: 4px 0; }
.weights-banner ul { margin: 4px 0 0 24px; padding: 0; }
/* Folder-tab score selector: pure-CSS radio + sibling-selector. The
   active tab visually merges into the panel below by matching the
   panel's background and hiding its bottom border. */
.scorer-tabs > input[type="radio"] { display: none; }
.scorer-tabs .tab-row { display: flex; border-bottom: 2px solid #5a92c8; margin: 20px 0 0 0; }
.scorer-tabs .tab-label { padding: 8px 18px; background: #2a2a2a; color: #b8b8b8;
                          border: 2px solid #3a3a3a; border-bottom: none;
                          border-radius: 6px 6px 0 0; margin-right: 4px;
                          margin-bottom: -2px; cursor: pointer; user-select: none;
                          font-weight: 500; }
.scorer-tabs .tab-label:hover { color: #cfe2f3; }
.scorer-tabs .tab-panel { display: none; padding: 16px 4px; }
.scorer-tabs > input#scorer-arc:checked ~ .tab-row label[for="scorer-arc"],
.scorer-tabs > input#scorer-weighted:checked ~ .tab-row label[for="scorer-weighted"],
.scorer-tabs > input#scorer-levels:checked ~ .tab-row label[for="scorer-levels"] {
    background: #1e1e1e; color: #ffffff; border-color: #5a92c8; border-bottom-color: #1e1e1e;
}
.scorer-tabs > input#scorer-arc:checked ~ #panel-arc,
.scorer-tabs > input#scorer-weighted:checked ~ #panel-weighted,
.scorer-tabs > input#scorer-levels:checked ~ #panel-levels { display: block; }
.scorer-tabs .scorer-desc { color: #b8b8b8; font-style: italic; margin: 4px 0 12px 0; max-width: 80ch; }
/* Two-level budget × scorer tab widget (comparison statistics).
   A panel shows only when its budget tab AND scorer tab are both
   checked — the standard two-radio-group CSS-only selection. The budget
   row is amber, the scorer row blue, to read as nested. */
.bs-tabs > input[type="radio"] { display: none; }
.bs-tabs .tab-row { display: flex; flex-wrap: wrap; }
.bs-tabs .budget-row { border-bottom: 2px solid #e0a060; margin: 20px 0 0 0; }
.bs-tabs .scorer-row { border-bottom: 2px solid #5a92c8; margin: 14px 0 0 0; }
.bs-tabs .tab-label { padding: 8px 18px; background: #2a2a2a; color: #b8b8b8;
                      border: 2px solid #3a3a3a; border-bottom: none;
                      border-radius: 6px 6px 0 0; margin-right: 4px;
                      margin-bottom: -2px; cursor: pointer; user-select: none; font-weight: 500; }
.bs-tabs .tab-label:hover { color: #cfe2f3; }
.bs-tabs .tab-panel { display: none; padding: 16px 4px; }
.bs-tabs > #bs-b-full:checked ~ .budget-row label[for="bs-b-full"],
.bs-tabs > #bs-b-capped:checked ~ .budget-row label[for="bs-b-capped"] {
    background: #1e1e1e; color: #ffffff; border-color: #e0a060; border-bottom-color: #1e1e1e;
}
.bs-tabs > #bs-s-arc:checked ~ .scorer-row label[for="bs-s-arc"],
.bs-tabs > #bs-s-weighted:checked ~ .scorer-row label[for="bs-s-weighted"],
.bs-tabs > #bs-s-levels:checked ~ .scorer-row label[for="bs-s-levels"] {
    background: #1e1e1e; color: #ffffff; border-color: #5a92c8; border-bottom-color: #1e1e1e;
}
.bs-tabs > #bs-b-full:checked ~ #bs-s-arc:checked ~ #bs-p-full-arc,
.bs-tabs > #bs-b-full:checked ~ #bs-s-weighted:checked ~ #bs-p-full-weighted,
.bs-tabs > #bs-b-full:checked ~ #bs-s-levels:checked ~ #bs-p-full-levels,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-arc:checked ~ #bs-p-capped-arc,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-weighted:checked ~ #bs-p-capped-weighted,
.bs-tabs > #bs-b-capped:checked ~ #bs-s-levels:checked ~ #bs-p-capped-levels { display: block; }
.bs-tabs .scorer-desc { color: #b8b8b8; font-style: italic; margin: 4px 0 12px 0; max-width: 80ch; }
.per-game-grid { display: flex; flex-direction: column; gap: 12px; margin: 12px 0; }
.per-game-card { display: flex; align-items: center; gap: 12px; padding: 12px;
                 background: #2a2a2a; border-radius: 4px; flex-wrap: wrap; }
.per-game-card img { max-width: 360px; height: auto; border-radius: 4px; }
.per-game-header { min-width: 160px; }
/* Pairwise p-value matrices (>2-run comparison). */
.summary-box h4 { margin: 14px 0 4px 0; }
table.pmatrix td { text-align: center; }
td.pval-sig { font-weight: bold; color: #9be8a8; }
td.pval-na { color: #888; }
td.pval-diag { color: #666; }
.pval-arrow { color: #e0a060; font-weight: bold; }
"""

# Stamped on the per-game wallclock plots only — the tokens plot has a clean
# budget interpretation (see ``_TOKENS_CURVE_NOTE``) but wallclock is
# trickier because games run in parallel during the benchmark.
_PLOT_DESIGN_NOTE = (
    '<div class="design-note">'
    "<strong>Design note:</strong> we still have to think about how to "
    "plot this exactly when we have some real benchmark data — games "
    "run in parallel during the benchmark, so wallclock per game is "
    "not just total wallclock / n_games."
    "</div>"
)

# Pinned next to the cross-game score-vs-job-wallclock plots: unlike the
# tokens plot it is not a per-game budget, and the reader would otherwise
# take it for one.
_JOB_WALLCLOCK_NOTE = (
    '<div class="curve-note">'
    "<p><strong>What this curve shows.</strong> The mean score across games "
    "<em>T</em> minutes after the job started: a game scores 0 until it "
    "starts, so games queued for a slot enter late. With every game weighted "
    "equally it is the <code>mean score</code> line of the run's stdout, drawn "
    "over time.</p>"
    "</div>"
)

# Pinned next to every *cross-game* score-vs-tokens plot. Skipped on
# per-game plots, where the curve already means "this single game's
# expected score at budget X" without further explanation.
_TOKENS_CURVE_NOTE = (
    '<div class="curve-note">'
    "<p><strong>What this curve shows.</strong> Reading the curve at <em>x = X</em> "
    "tells you the expected score per "
    "game when the method is run at an average per-game budget of <em>X</em> "
    "tokens. Slow games may use more, fast games less; the average works "
    "out to <em>X</em>. </p>"
    "<p><strong>It may go down, and that is not a plotting bug.</strong> A "
    "solver can replace its answer, and a task it had right at one spend can "
    "be wrong at the next. Individual tasks doing that washes out across a "
    "set this size, so a visible dip means the solver is systematically "
    "talking itself out of correct answers &mdash; which is worth knowing "
    "and is otherwise invisible.</p>"
    "</div>"
)


def _is_in_flight(benchmark: taaf.benchmark.Benchmark) -> bool:
    """A benchmark is in-flight if ``run()`` started but hasn't finished yet
    (the periodic save loop generates diagnostics mid-run). After
    teardown ``end_time`` is set in ``Benchmark.run``'s ``finally`` block, so
    ``end_time is None`` while ``start_time is not None`` is the unambiguous
    in-flight signal.
    """
    return benchmark.start_time is not None and benchmark.end_time is None


def _in_flight_banner_html(benchmark: taaf.benchmark.Benchmark) -> str:
    """Banner warning that these numbers are a partial snapshot."""
    if not _is_in_flight(benchmark):
        return ""
    started = _format_timestamp(benchmark.start_time) if benchmark.start_time else "?"
    written = _format_timestamp(datetime.now())
    return (
        '<div class="in-flight-banner">'
        "⏳ <strong>Run in progress.</strong> This page was generated by the "
        f"periodic save loop while the benchmark is still executing (started <code>{started}</code>, "
        f"this snapshot written <code>{written}</code>). "
        "Numbers, plots, and answer sheets reflect a partial snapshot — they will be "
        "rewritten when the run finishes."
        "</div>"
    )


def _html_page(title: str, body: str) -> str:
    """Wrap a body in the shared page shell."""
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        f"<style>{_CSS}</style></head>\n<body>\n{body}\n</body></html>\n"
    )


def _safe(s: str) -> str:
    """A filename-safe form of an arbitrary id."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def _game_name_html(run: taaf.game.GameRun, thumb_px: int = THUMB_DEFAULT_PX) -> str:
    """A game id with its mini-figure beside it."""
    thumb = _thumbnail_html(run, width_px=thumb_px)
    return f'<span class="game-name">{thumb}<code>{escape(run.game_id)}</code></span>'


SHEET_GRID_PX = 96

#: Bump when the answer sheet's page changes, so the sheets an earlier version
#: wrote into a job dir are rewritten on its next render (``_answer_sheet_key``).
_ANSWER_SHEET_VERSION = 1

#: The per-sheet keys the sheets in ``answers/`` were last written with, beside
#: the per-game card cache.
_ANSWER_SHEET_INDEX = ".answer_sheets.pkl"


def _answer_sheet_key(run: taaf.game.GameRun, solver_label: str) -> str:
    """Everything a finished run's sheet is drawn from, for knowing whether
    the sheet on disk is still the one to write.

    The task is fixed by the sheet's (game, pass); the last submission by
    ``len(history)``, since records are frozen and only ever appended; the
    verdicts by ``result``, also frozen; and the link to the solver's own
    analysis may be set after the run finishes.
    """
    return "|".join(
        (
            str(_ANSWER_SHEET_VERSION),
            solver_label,
            run.state,
            str(len(run.history)),
            repr(run.result),
            run.solver_analysis_html or "",
        )
    )


def _load_answer_sheet_index(path: Path) -> dict[str, str]:
    """Load ``{sheet stem: key}``; empty on a miss, as the card cache does."""
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            index: Any = pickle.load(f)
    except Exception:
        # Corrupted / incompatible: every sheet is rewritten, which is correct.
        return {}
    if not isinstance(index, dict):
        return {}
    return cast("dict[str, str]", index)


def _answer_sheet_html(run: taaf.game.GameRun, title: str, solver_label: str) -> str:
    """One (game, pass) answer sheet, as a page.

    The whole task on one static page: every demonstration pair as
    question and answer, then every pair the solver had to answer as
    question, both attempts, and the truth. This is the page that says what
    the solver actually claimed, which a recording of a board that never
    moves could not.
    """
    spec = run.spec
    assert spec is not None, "an answer sheet needs the run's spec"
    analysis_html = ""
    if run.solver_analysis_html:
        # The sheet lives at {job_dir}/answers/{stem}.html and the analysis
        # path is relative to job_dir.
        analysis_html = f'<p><a href="../{run.solver_analysis_html}">Detailed analysis by solver &rarr;</a></p>'

    given_rows = "".join(
        f"<tr><td>{i}</td><td>{_grid_img_html(pair.question, SHEET_GRID_PX, 'question')}</td>"
        f"<td>{_grid_img_html(pair.answer, SHEET_GRID_PX, 'answer') if pair.answer else ''}</td></tr>"
        for i, pair in enumerate(spec.given)
    )
    submission = run.history[-1].submission if run.history else None
    answer_rows: list[str] = []
    for i, pair in enumerate(spec.to_answer):
        attempts = submission.attempts[i] if submission is not None and i < len(submission.attempts) else ()
        pair_result = run.result.pairs[i] if run.result is not None and i < len(run.result.pairs) else None
        cells = [f"<td>{i}</td>", f"<td>{_grid_img_html(pair.question, SHEET_GRID_PX, 'question')}</td>"]
        for k in range(taaf.game.N_ATTEMPTS):
            if k < len(attempts):
                hit = pair_result is not None and pair_result.matched_attempt == k
                mark = ' <span style="color:#4caf50">&#10003;</span>' if hit else ""
                cells.append(f"<td>{_grid_img_html(attempts[k], SHEET_GRID_PX, f'attempt {k + 1}')}{mark}</td>")
            else:
                cells.append("<td>&mdash;</td>")
        truth = (
            _grid_img_html(pair.answer, SHEET_GRID_PX, "correct") if pair.answer is not None else "<em>withheld</em>"
        )
        if submission is None or pair_result is None:
            verdict = "&mdash;"
        elif pair_result.correct is None:
            verdict = "<em>unscored</em>"
        elif pair_result.correct:
            verdict = '<span style="color:#4caf50">correct</span>'
        else:
            verdict = '<span style="color:#e57373">wrong</span>'
        cells.append(f"<td>{truth}</td><td>{verdict}</td>")
        answer_rows.append(f"<tr>{''.join(cells)}</tr>")

    attempt_headers = "".join(f"<th>attempt {k + 1}</th>" for k in range(taaf.game.N_ATTEMPTS))
    body = (
        '<p><a href="../diagnostics.html">&larr; back to diagnostics</a></p>'
        f"<h1>{escape(title)}</h1>"
        f"<p>Solver: <code>{escape(solver_label)}</code> &middot; state <code>{escape(run.state)}</code> "
        f"&middot; correct {_correct_cell(run)}</p>"
        f"{analysis_html}"
        "<h2>Given pairs</h2>"
        f"<table><tr><th>#</th><th>question</th><th>answer</th></tr>{given_rows}</table>"
        "<h2>Answers</h2>"
        f"<table><tr><th>#</th><th>question</th>{attempt_headers}<th>correct</th><th>verdict</th></tr>"
        f"{''.join(answer_rows)}</table>"
    )
    return _html_page(title, body)


# --- Public: per-run HTML ---------------------------------------------------


def generate_run_html(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> None:
    """Per-run HTML at ``out_path``, with one answer sheet per (game, pass)
    under ``{out_path.parent}/answers/``: ``render_run_html``, written."""
    taaf.support.write_files(render_run_html(benchmark, out_path))


def render_run_html(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> list[taaf.support.FileWrite]:
    """What ``generate_run_html`` writes, rendered but not written, in the
    order to write it.

    The periodic save loop renders here, on the event loop, where the runs the
    solver keeps changing hold still, and writes on a thread: writing is the
    expensive half on a network filesystem, where every file costs several
    round trips to the server.

    A sheet is written when it is new or its run changed since the last
    render (``_answer_sheet_key``), so the loop writes the games that finished
    since its last save rather than every sheet again. The page's tab widget
    selects between Official ARC and Tasks solved for the top-of-page plots;
    per-game cards and the per-pass-per-game table always render under ARC.
    In-flight saves build only the ARC panel.
    """
    writes: list[taaf.support.FileWrite] = []
    n_games = len(_game_ids(benchmark))

    # ---- Fixed parts (rendered once, always under ARC) --------------------

    per_game_scatter_b64 = _render_per_game_score_vs_tokens_scatter_png(
        benchmark, title=f"{benchmark.label} — per-game score vs tokens"
    )

    tokens_vs_wall_b64 = _render_tokens_vs_wallclock_png(
        [benchmark],
        title=f"{benchmark.label} — total generated tokens vs job wallclock",
    )

    # One static answer sheet per (game, pass): what was asked, what the
    # solver said, and what was right. Rendered and written only where the
    # sheet on disk is missing or was drawn from a different state of its run.
    # Rewriting every finished game's sheet at every save was most of a
    # periodic save on NFS, where each file is several round trips to the
    # server, and grew with every game that finished.
    sheets_dir = out_path.parent / "answers"
    index_path = out_path.parent / _ANSWER_SHEET_INDEX
    written = _load_answer_sheet_index(index_path)
    # One directory listing rather than a lookup per sheet: a sheet the index
    # remembers but the directory lacks (deleted, or a copied job dir) is
    # written again.
    on_disk = {entry.name for entry in os.scandir(sheets_dir)} if sheets_dir.is_dir() else set[str]()
    index: dict[str, str] = {}
    sheet_rels: dict[tuple[str, int], str | None] = {}
    for p in range(benchmark.n_passes):
        for run in _pass_runs(benchmark, p):
            stem = f"g{_safe(run.game_id)}_p{p}"
            if run.spec is None or run.state in ("not_started", "playing"):
                sheet_rels[(run.game_id, p)] = None
                continue
            key = _answer_sheet_key(run, benchmark.solver_label)
            if written.get(stem) != key or f"{stem}.html" not in on_disk:
                writes.append(
                    taaf.support.FileWrite(
                        sheets_dir / f"{stem}.html",
                        _answer_sheet_html(run, title=f"{run.game_id} — pass {p}", solver_label=benchmark.solver_label),
                    )
                )
            index[stem] = key
            sheet_rels[(run.game_id, p)] = f"answers/{stem}.html"
    # After the sheets it describes: a write that fails partway leaves the
    # previous index, and the next render redoes what was lost.
    if index != written:
        writes.append(
            taaf.support.FileWrite(index_path, pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL), atomic=True)
        )

    # Game-major rows so single-game scanning stays local. Score column
    # is always ARC, independent of which scorer tab is active.
    rows = [
        "<tr><th>game</th><th>pass</th><th>state</th><th>score</th>"
        "<th>correct</th><th>tokens</th><th>answers</th><th>note</th></tr>"
    ]
    for game_id in _game_ids(benchmark):
        for p, run in enumerate(_game_runs_by_id(benchmark, game_id)):
            score = _arc_score(run)
            tokens = total_tokens(run)
            sheet_rel = sheet_rels.get((run.game_id, p))
            thumb = _thumbnail_html(run, width_px=THUMB_FINAL_PX)
            cell = f'<a href="{sheet_rel}">{thumb or "view"}</a>' if sheet_rel else "—"
            note_html = escape(run.solver_note) if run.solver_note else ""
            rows.append(
                f"<tr><td>{_game_name_html(run)}</td>"
                f"<td>{p}</td>"
                f"{_state_cell_html(run.state)}"
                f"<td>{'?' if run.current_score() is None else f'{score:.2f}'}</td>"
                f"<td>{_correct_cell(run)}</td>"
                f"<td>{tokens}</td>"
                f"<td>{cell}</td>"
                f"<td>{note_html}</td></tr>"
            )
    table_html = "<table>" + "".join(rows) + "</table>"

    git_status_html = _git_status_block_html(benchmark)

    per_game_grid_html = _per_game_grid_for_run_html(benchmark, writes=writes)

    # ---- Per-scorer tab panels --------------------------------------------
    # In-flight: only ARC, to keep the periodic save loop cheap (variant
    # scores recompute from scratch each save).
    scorers_to_render = [ARC_SCORER] if _is_in_flight(benchmark) else SCORERS
    weights_by_id = weights_by_game_id(benchmark)

    panels: list[tuple[Scorer, str]] = []
    for scorer in scorers_to_render:
        tokens_curve = _pass_curve(benchmark.game_runs, TOKEN_AXIS, scorer=scorer, weights_by_game_id=weights_by_id)
        wall_curve = _benchmark_score_vs_job_wallclock_minutes(
            benchmark, scorer=scorer, weights_by_game_id=weights_by_id
        )
        # Per-pass overlays under each main curve so cross-pass variance
        # is visible. Skipped at n_passes == 1.
        per_pass_tokens: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        per_pass_wall: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        if benchmark.n_passes > 1:
            for p in range(benchmark.n_passes):
                pass_runs = _pass_runs(benchmark, p)
                per_pass_tokens.append(
                    _pass_curve(pass_runs, TOKEN_AXIS, scorer=scorer, weights_by_game_id=weights_by_id)
                )
                per_pass_wall.append(
                    _benchmark_score_vs_job_wallclock_minutes(
                        benchmark, pass_runs, scorer=scorer, weights_by_game_id=weights_by_id
                    )
                )
        tokens_b64 = _render_curves_png(
            [tokens_curve],
            x_label="average tokens per game",
            y_label=scorer.y_axis_label,
            title=f"{benchmark.label} — {scorer.label} vs total tokens",
            overlay_curves=per_pass_tokens or None,
        )
        wall_b64 = _render_curves_png(
            [wall_curve],
            x_label="job wallclock (min)",
            y_label=scorer.y_axis_label,
            title=f"{benchmark.label} — {scorer.label} vs job wallclock",
            overlay_curves=per_pass_wall or None,
        )
        panel_inner = (
            f"{_per_run_stats_box([benchmark], scorer=scorer)}"
            "<h2>Score vs cumulative tokens</h2>"
            f"{_TOKENS_CURVE_NOTE}"
            f'<img src="data:image/png;base64,{tokens_b64}" alt="score vs tokens">'
            "<h2>Score vs job wallclock</h2>"
            f"{_JOB_WALLCLOCK_NOTE}"
            f'<img src="data:image/png;base64,{wall_b64}" alt="score vs job wallclock">'
        )
        panels.append((scorer, panel_inner))

    tabs_html = _scorer_tabs_html(panels)
    # Skip the boundary callout when there's nothing to contrast (in-flight
    # or single-scorer page) — it would be misleading.
    remainder_note = _VARIANT_REMAINDER_NOTE if len(scorers_to_render) > 1 else ""

    body = (
        f"<h1>{escape(benchmark.label or 'benchmark')}</h1>"
        f"{_in_flight_banner_html(benchmark)}"
        '<p class="intro">Diagnostics for one TAAF benchmark run — a single solver played each '
        "game over a fixed number of passes. Plots track score-vs-cost across the run; the "
        "table lists every (game, pass) cell, with the answers cell linking to a full render of that "
        "play-through.</p>"
        '<div class="summary-box">'
        f"<p>Solver: <code>{escape(benchmark.solver_label)}</code></p>"
        f"<p>{n_games} game{'s' if n_games != 1 else ''} ×"
        f" {benchmark.n_passes} pass{'es' if benchmark.n_passes != 1 else ''}"
        f" = {len(benchmark.game_runs)} runs</p>"
        f"{_timestamps_block_html(benchmark.start_time, benchmark.end_time, with_end=True)}"
        f"{git_status_html}"
        "</div>"
        f"{_weights_banner_html(benchmark)}"
        f"{tabs_html}"
        f"{remainder_note}"
        "<h2>Total generated tokens vs job wallclock</h2>"
        f'<img src="data:image/png;base64,{tokens_vs_wall_b64}" alt="total tokens vs job wallclock">'
        "<h2>Per-game score vs tokens</h2>"
        f'<img src="data:image/png;base64,{per_game_scatter_b64}" alt="per-game score vs tokens scatter">'
        "<h2>Per-game</h2>"
        f"{per_game_grid_html}"
        "<h2>Per-pass per-game results</h2>"
        f"{table_html}"
    )
    # Last: the page links the sheets, which land first.
    writes.append(taaf.support.FileWrite(out_path, _html_page(benchmark.label or "benchmark", body)))
    return writes


# --- Public: per-run summary ------------------------------------------------


def run_summary_text(benchmark: taaf.benchmark.Benchmark) -> str:
    """Build the Slack-friendly text summary. Also sourced by
    ``Benchmark.run``'s end-of-run stdout print.
    """
    scores = [_arc_score(r) for r in benchmark.game_runs]
    won = sum(1 for r in benchmark.game_runs if r.state == "won")
    # A crashed run scores 0 like a run that played and lost, so the score
    # figures alone cannot tell "the solver is weak" from "every run
    # raised". Tallied here, next to the count it qualifies.
    by_state: dict[str, int] = {}
    for run in benchmark.game_runs:
        by_state[run.state] = by_state.get(run.state, 0) + 1
    crashed = by_state.get("crashed", 0)
    total_actions = sum(_count_actions(r) for r in benchmark.game_runs)
    tokens_total = sum(total_tokens(r) for r in benchmark.game_runs)
    total_wall = sum(_total_wallclock(r) for r in benchmark.game_runs)
    elapsed_seconds = _benchmark_elapsed_seconds(benchmark)
    generated_tokens_per_second = tokens_total / elapsed_seconds if elapsed_seconds and elapsed_seconds > 0 else None

    lines: list[str] = [
        f"benchmark: {benchmark.label}",
        f"solver:    {benchmark.solver_label}",
        f"games:     {len(_game_ids(benchmark))}",
        f"passes:    {benchmark.n_passes}",
        f"runs:      {len(benchmark.game_runs)} (won: {won})",
        "states:    " + ", ".join(f"{count} {state}" for state, count in sorted(by_state.items())),
    ]
    if benchmark.start_time is not None:
        lines.append(f"started:   {_format_timestamp(benchmark.start_time)}")
    if benchmark.end_time is not None and benchmark.start_time is not None:
        lines.append(f"ended:     {_format_timestamp(benchmark.end_time)}")
        lines.append(f"duration:  {_format_duration(benchmark.end_time - benchmark.start_time)}")
    elif benchmark.start_time is not None:
        lines.append("ended:     in progress")
    if benchmark.stop_reason is not None:
        lines.append(f"stopped:   {benchmark.stop_reason}")
    lines.extend(
        [
            f"mean score:    {statistics.mean(scores):.2f}" if scores else "mean score:    —",
            f"median score:  {statistics.median(scores):.2f}" if scores else "median score:  —",
            f"total actions: {total_actions}",
            f"total tokens:  {tokens_total}",
            (
                f"generated tokens/sec: {generated_tokens_per_second:.2f} (job wallclock)"
                if generated_tokens_per_second is not None
                else "generated tokens/sec: —"
            ),
            f"total wallclock: {total_wall:.1f}s",
        ]
    )
    # Only for crashed: a run cancelled by its wall budget played, it just
    # did not finish, and saying otherwise would fire on every budgeted run.
    if benchmark.game_runs and crashed == len(benchmark.game_runs):
        lines.append("*** every run CRASHED: the scores below are exceptions, not play ***")
    lines.extend(
        [
            "",
            "per-game (mean across passes):",
        ]
    )
    for g in _game_ids(benchmark):
        runs = _game_runs_by_id(benchmark, g)
        gs = [_arc_score(r) for r in runs]
        tokens = sum(total_tokens(r) for r in runs) // max(len(runs), 1)
        n_pairs = _n_to_answer(runs[0]) if runs else 0
        results = [r.current_result() for r in runs]
        correct_avg = sum(x.n_correct if x is not None and x.scored else 0 for x in results) / max(len(runs), 1)
        unscored = any(r.current_score() is None for r in runs)
        score_str = "unscored" if unscored else (f"{statistics.mean(gs):.2f}" if gs else "—")
        correct_str = "?" if unscored else f"{correct_avg:.1f}"
        lines.append(f"  {g}: score={score_str}, correct={correct_str}/{n_pairs}, tokens={tokens}")
    return "\n".join(lines) + "\n"


def generate_run_summary_txt(benchmark: taaf.benchmark.Benchmark, out_path: Path) -> None:
    """Write the brief text summary to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(run_summary_text(benchmark))


# --- Public: comparison HTML ------------------------------------------------


def _harmonize_for_comparison(
    benchmarks: list[taaf.benchmark.Benchmark],
    labels: list[str],
) -> tuple[list[str], set[str]]:
    """Mutate ``benchmarks`` in place so a comparison page can be rendered.

    - If game-id sets differ, restrict each benchmark to the intersection
      (drop game_runs / game_weights / games slots accordingly, preserving
      the pass-major layout).
    - If resolved ``{game_id: weight}`` maps differ, override every
      non-first benchmark's ``game_weights`` with the first benchmark's
      values (mapped by game_id, so slot-order differences are fine).

    Returns ``(notices, intersection)``. ``notices`` is a list of HTML
    snippets describing each mutation applied (empty when nothing
    changed). ``intersection`` is the resolved set of common game ids;
    an empty set signals the caller to skip rendering entirely.
    """
    notices: list[str] = []
    game_id_sets: list[set[str]] = [set(_game_ids(b)) for b in benchmarks]
    # ``set.intersection(*…)`` unpacked-call loses element type through pyright's
    # variadic stub; iterate so the result stays typed as ``set[str]``.
    intersection: set[str] = set[str]()
    if game_id_sets:
        intersection = set(game_id_sets[0])
        for s in game_id_sets[1:]:
            intersection &= s
    if any(s != intersection for s in game_id_sets):
        parts: list[str] = []
        for lbl, s in zip(labels, game_id_sets):
            dropped = sorted(s - intersection)
            if not dropped:
                continue
            shown = ", ".join(escape(g) for g in dropped[:5])
            if len(dropped) > 5:
                shown += f", … ({len(dropped) - 5} more)"
            parts.append(f"<code>{escape(lbl)}</code> dropped {len(dropped)} ({shown})")
        notices.append(
            "<strong>Game sets differ across runs.</strong> Restricted to the intersection "
            f"({len(intersection)} game{'s' if len(intersection) != 1 else ''}). " + "; ".join(parts) + "."
        )
        for b in benchmarks:
            if not b.game_runs:
                continue
            canonical = [r.game_id for r in _pass_runs(b, 0)]
            keep = [i for i, gid in enumerate(canonical) if gid in intersection]
            kept_ids = {canonical[i] for i in keep}
            b.game_runs = [r for r in b.game_runs if r.game_id in kept_ids]
            if b.game_weights is not None:
                b.game_weights = [b.game_weights[i] for i in keep]
            if len(b.games) == len(canonical):
                b.games = [b.games[i] for i in keep]
    if not intersection:
        return notices, intersection
    # Weights are resolved via ``{game_id: weight}`` so different
    # ``Benchmark.games`` orderings compare semantically. ``None``
    # resolves to all-1.0 — equal to an explicit all-1.0 list.
    weights_maps = [weights_by_game_id(b) for b in benchmarks]
    if any(w != weights_maps[0] for w in weights_maps[1:]):
        notices.append(
            f"<strong>Game weights differ across runs.</strong> Applied "
            f"<code>{escape(labels[0])}</code>'s <code>game_weights</code> to every run."
        )
        first_weights = weights_maps[0]
        for b in benchmarks[1:]:
            if not b.game_runs:
                continue
            slot_gids = [r.game_id for r in _pass_runs(b, 0)]
            b.game_weights = [first_weights.get(g, 1.0) for g in slot_gids]
    return notices, intersection


def _harmonization_banner_html(notices: list[str]) -> str:
    """Yellow banner above the comparison body listing any in-place
    harmonization the renderer applied to the input benchmarks."""
    if not notices:
        return ""
    items = "".join(f"<li>{n}</li>" for n in notices)
    return (
        f'<div class="weights-banner"><p><strong>Inputs harmonized for comparison.</strong></p><ul>{items}</ul></div>'
    )


def generate_comparison_html(benchmarks: list[taaf.benchmark.Benchmark], out_dir: Path) -> None:
    """Write the comparison tree at ``out_dir``
    (``index.html`` plus ``per_game/<game>.html`` drill-downs).

    Refuses in-flight inputs — cross-run statistics are meaningless on
    partial snapshots. Inputs are harmonized in place via
    ``_harmonize_for_comparison`` (intersection of game-id sets;
    weights unified to the first benchmark). Both surface in a banner.
    Empty intersection raises rather than leaving a stale index.html.
    """
    if len(benchmarks) < 2:
        raise ValueError("generate_comparison_html requires at least 2 benchmarks")
    in_flight = [b.label or f"run {i}" for i, b in enumerate(benchmarks) if _is_in_flight(b)]
    if in_flight:
        raise ValueError(
            f"cannot generate comparison: {len(in_flight)} benchmark(s) still in-flight: "
            f"{in_flight}. Wait for run() to finish (end_time gets set in the teardown)."
        )
    labels = [b.label or f"run {i}" for i, b in enumerate(benchmarks)]
    harmonize_notices, intersection = _harmonize_for_comparison(benchmarks, labels)
    if not intersection:
        # Raise rather than no-op: a quiet return would leave a stale
        # index.html from a previous compare reading as fresh output.
        raise ValueError(
            f"generate_comparison_html: empty intersection of game-id sets across "
            f"{len(benchmarks)} benchmarks — nothing to compare. Adjust the input "
            f"benchmarks so at least one game_id is common to all of them."
        )
    common_games = sorted(intersection)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_game_dir = out_dir / "per_game"
    per_game_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fixed parts (rendered once, always under ARC) --------------------

    # Per-game cards (initial-state thumbnail + tokens / wallclock
    # plots) linking to the per-game drill-down.
    per_game_section = _per_game_grid_html(benchmarks, common_games, labels)

    # Top-of-page links to each run's per-run diagnostics. Skipped when
    # a benchmark has no job_dir or its diagnostics.html isn't on disk
    # yet (comparison generated standalone before the per-run save).
    per_run_links: list[str] = []
    for b, lbl in zip(benchmarks, labels):
        if b.job_dir is None:
            continue
        target = b.job_dir / "diagnostics.html"
        if not target.exists():
            continue
        rel = os.path.relpath(target, start=out_dir)
        per_run_links.append(f'<a href="{rel}"><code>{escape(lbl)}</code></a>')
    per_run_links_html = f"<p>Per-run details: {' · '.join(per_run_links)}</p>" if per_run_links else ""

    runs_lines: list[str] = []
    for b in benchmarks:
        passes = f"{b.n_passes} pass{'es' if b.n_passes != 1 else ''}"
        when = f" — started {_format_timestamp(b.start_time)}" if b.start_time else ""
        runs_lines.append(f"<li><code>{escape(b.label)}</code> ({passes}){when}</li>")
    runs_summary = "<ul>" + "".join(runs_lines) + "</ul>"

    # Per-game pages — always rendered under ARC, regardless of which
    # comparison index variant the reader is on.
    for g in common_games:
        _write_per_game_page(g, benchmarks, per_game_dir / f"{_safe(g)}.html")

    # ---- Per-scorer tab panels --------------------------------------------
    # All three scorers always render — in-flight inputs are refused
    # above.

    # Two statistics variants: runs scored at their endpoints, and
    # re-scored at a shared per-game token budget so a run that merely spent
    # more tokens doesn't look better for it. The budget is
    # scorer-independent, so it's computed once. Rendered as a budget ×
    # scorer tab grid (budget tabs above scorer tabs).
    budget = _min_tokens_per_game_budget(benchmarks)
    ceilings = ceilings_by_run_id(benchmarks, budget) if budget > 0 else {}
    budget_modes: list[tuple[str, str, bool, str]] = [
        (
            "full",
            "Full runs (max tokens)",
            False,
            "Each run scored where it actually ended, at its own per-game token budget.",
        )
    ]
    if budget > 0:
        budget_modes.append(
            (
                "capped",
                f"Common budget ({budget:,.0f} tok/game)",
                True,
                (
                    f"Every (game, pass) re-scored at a shared per-game budget of {budget:,.0f} generated "
                    "tokens — the smallest run's mean tokens per game, the rightmost point all runs' "
                    "score-vs-tokens curves share. Modeled as a parallel cutoff WITHIN each run: its games "
                    "spend tokens together and freeze when that run's average game has spent the budget, so "
                    "cheaper games keep their final score while more expensive ones are capped. Every run "
                    "therefore reads at the same tokens per game — which is what its score-vs-tokens curve "
                    "above shows at this budget."
                ),
            )
        )

    scorer_panels: list[tuple[Scorer, dict[str, str]]] = []
    for scorer in SCORERS:
        # Pooled curves are budget-independent (they span the full range),
        # so render once per scorer and reuse in every budget panel.
        token_curves = [
            _pass_curve(b.game_runs, TOKEN_AXIS, scorer=scorer, weights_by_game_id=weights_by_game_id(b))
            for b in benchmarks
        ]
        wall_curves = [
            _benchmark_score_vs_job_wallclock_minutes(b, scorer=scorer, weights_by_game_id=weights_by_game_id(b))
            for b in benchmarks
        ]
        tokens_b64 = _render_curves_png(
            token_curves,
            x_label="average tokens per game",
            y_label=scorer.y_axis_label,
            title=f"comparison — {scorer.label} vs total tokens",
            labels=labels,
            palette=RUN_COLORS,
        )
        wall_b64 = _render_curves_png(
            wall_curves,
            x_label="job wallclock (min)",
            y_label=scorer.y_axis_label,
            title=f"comparison — {scorer.label} vs job wallclock",
            labels=labels,
            palette=RUN_COLORS,
        )
        plots_html = (
            "<h2>Score vs cumulative tokens</h2>"
            f"{_TOKENS_CURVE_NOTE}"
            f'<img src="data:image/png;base64,{tokens_b64}" alt="comparison tokens">'
            "<h2>Score vs job wallclock</h2>"
            f"{_JOB_WALLCLOCK_NOTE}"
            f'<img src="data:image/png;base64,{wall_b64}" alt="comparison job wallclock">'
        )

        by_budget: dict[str, str] = {}
        for mode_key, _mode_label, is_capped, note in budget_modes:
            score_fn = _capped_score_fn(scorer, ceilings) if is_capped else None
            per_run_html = _per_run_stats_box(benchmarks, scorer=scorer, score_fn=score_fn)
            stats_box = _comparison_stats_box(
                benchmarks,
                labels,
                scorer=scorer,
                score_fn=score_fn,
                heading="Score comparison",
                note=note,
            )
            scatter_html = ""
            if len(benchmarks) == 2:
                per_game_a = _per_game_mean_scores(benchmarks[0], scorer=scorer, score_fn=score_fn)
                per_game_b = _per_game_mean_scores(benchmarks[1], scorer=scorer, score_fn=score_fn)
                scatter_b64 = _render_scatter_png(
                    per_game_a,
                    per_game_b,
                    labels[0],
                    labels[1],
                    f"head-to-head per-game mean — {scorer.label}",
                )
                scatter_html = (
                    f'<h2>Head-to-head</h2><img src="data:image/png;base64,{scatter_b64}" alt="head-to-head scatter">'
                )
            by_budget[mode_key] = f"{per_run_html}{stats_box}{plots_html}{scatter_html}"
        scorer_panels.append((scorer, by_budget))

    tabs_html = _budget_scorer_tabs_html(
        scorer_panels,
        budget_labels=[(k, lbl) for k, lbl, _, _ in budget_modes],
    )

    tokens_vs_wall_b64 = _render_tokens_vs_wallclock_png(
        benchmarks,
        title="comparison — total generated tokens vs job wallclock",
        labels=labels,
        palette=RUN_COLORS,
    )

    body = (
        f"<h1>Comparison: {' vs '.join(escape(label) for label in labels)}</h1>"
        '<p class="intro">Side-by-side comparison of multiple TAAF benchmark runs on the same set '
        "of games. The three score tests appear under budget tabs (full runs vs a shared per-game "
        "token budget — the smallest run's) above the score-variant tabs, as a head-to-head box for "
        "two runs or NxN matrices of pairwise p-values for more. With exactly two runs you also get a "
        "head-to-head scatter. Click a per-game thumbnail to drill into how every run did on that "
        "single game.</p>"
        f'<div class="summary-box"><p>Runs:</p>{runs_summary}'
        f"<p>Games: {len(common_games)}</p></div>"
        f"{per_run_links_html}"
        f"{_harmonization_banner_html(harmonize_notices)}"
        f"{_comparison_weights_banner_html(benchmarks, labels)}"
        f"{tabs_html}"
        f"{_VARIANT_REMAINDER_NOTE}"
        "<h2>Total generated tokens vs job wallclock</h2>"
        f'<img src="data:image/png;base64,{tokens_vs_wall_b64}" alt="total tokens vs job wallclock">'
        f"{per_game_section}"
    )
    (out_dir / "index.html").write_text(_html_page("Comparison", body))


def _per_game_run_curve(
    bm: taaf.benchmark.Benchmark,
    game_id: str,
    axis: CostAxis,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Single-game pooled-pass curve for one benchmark.

    Treats every pass of ``game_id`` as one independent lane and runs them
    through ``_pass_curve``. The result is "expected (cost, score) for this
    game at the given budget", which is the per-game-flavoured analogue of
    the cross-game pooled curves on the run / comparison pages.
    """
    runs = _game_runs_by_id(bm, game_id)
    if not runs:
        return np.array([0.0]), np.array([0.0])
    return _pass_curve(runs, axis)


def _per_game_plots_b64(
    benchmarks: list[taaf.benchmark.Benchmark],
    game_id: str,
    labels: list[str],
    *,
    figsize: tuple[float, float] = (4.5, 3),
) -> tuple[str, str]:
    """Per-game (tokens, wallclock) plots — one pooled curve per run."""
    token_curves = [_per_game_run_curve(b, game_id, TOKEN_AXIS) for b in benchmarks]
    wall_curves = [_per_game_run_curve(b, game_id, WALLCLOCK_AXIS) for b in benchmarks]
    tokens = _render_curves_png(
        token_curves,
        x_label="average tokens for this game",
        y_label="average score for this game",
        title=f"{game_id} — score vs tokens",
        labels=labels,
        palette=RUN_COLORS,
        figsize=figsize,
    )
    wall = _render_curves_png(
        wall_curves,
        x_label="average wallclock for this game (s)",
        y_label="average score for this game",
        title=f"{game_id} — score vs wallclock",
        labels=labels,
        palette=RUN_COLORS,
        figsize=figsize,
    )
    return tokens, wall


_PER_GAME_CARD_CACHE_VERSION = 3
"""Stamped into the cache key; bump when the rendering of a per-game
card changes so old caches are invalidated rather than served stale."""


def _per_game_card_key(g: str, runs: list[taaf.game.GameRun], in_flight: bool) -> str:
    """Stable cache key — the card output is a pure function of these.
    The in-flight flag is part of the key because in-flight cards skip
    the per-pass overlay; without it, the post-run regen would re-use
    the overlay-free version.
    """
    parts = [str(_PER_GAME_CARD_CACHE_VERSION), g, "if" if in_flight else "done"]
    for r in runs:
        parts.append(f"{r.state}:{len(r.history)}:{r.final_score}")
    return "|".join(parts)


def _load_per_game_card_cache(job_dir: Path | None) -> dict[str, tuple[str, str]]:
    """Load `{game_id: (key, html)}` from disk; empty dict on miss."""
    if job_dir is None:
        return {}
    path = job_dir / ".per_game_cards.pkl"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            cached_obj: Any = pickle.load(f)
    except Exception:
        # Corrupted / incompatible — caller will overwrite.
        return {}
    if not isinstance(cached_obj, dict):
        return {}
    return cast("dict[str, tuple[str, str]]", cached_obj)


def _per_game_card_cache_write(
    job_dir: Path | None, cache: dict[str, tuple[str, str]]
) -> taaf.support.FileWrite | None:
    """The write that persists rendered per-game cards, so a periodic save
    does not redraw every finished game each sweep."""
    if job_dir is None:
        return None
    return taaf.support.FileWrite(job_dir / ".per_game_cards.pkl", pickle.dumps(cache), atomic=True)


def _per_game_grid_for_run_html(
    benchmark: taaf.benchmark.Benchmark,
    *,
    writes: list[taaf.support.FileWrite],
    figsize: tuple[float, float] = (4.5, 3),
) -> str:
    """Per-game cards for the per-run page: thumbnail + (tokens,
    wallclock) plots. One pooled curve per game.

    Cached on disk at ``{job_dir}/.per_game_cards.pkl``, keyed on
    per-pass ``(state, len(history), final_score)``.
    A finished pass freezes its key so each card renders once and is
    reused on every subsequent regen. The cache's write is appended to
    ``writes`` rather than made here (``render_run_html``).
    """
    in_flight = _is_in_flight(benchmark)
    cache = _load_per_game_card_cache(benchmark.job_dir)
    new_cache: dict[str, tuple[str, str]] = {}
    cards: list[str] = []
    for g in _game_ids(benchmark):
        runs = _game_runs_by_id(benchmark, g)
        if not runs:
            continue
        key = _per_game_card_key(g, runs, in_flight)
        cached = cache.get(g)
        if cached is not None and cached[0] == key:
            cards.append(cached[1])
            new_cache[g] = cached
            continue
        thumb = _thumbnail_html(runs[0], width_px=THUMB_FINAL_PX)
        # Per-pass overlays: skipped in-flight (cheap periodic loop)
        # and when n_passes == 1 (would duplicate the pooled curve).
        per_pass_tokens: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        per_pass_wall: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        if not in_flight and len(runs) > 1:
            for r in runs:
                per_pass_tokens.append(_pass_curve([r], TOKEN_AXIS))
                per_pass_wall.append(_pass_curve([r], WALLCLOCK_AXIS))
        tokens_b64 = _render_curves_png(
            [_pass_curve(runs, TOKEN_AXIS)],
            x_label="average tokens for this game",
            y_label="average score for this game",
            title=f"{g} — score vs tokens",
            figsize=figsize,
            overlay_curves=per_pass_tokens or None,
        )
        wall_b64 = _render_curves_png(
            [_pass_curve(runs, WALLCLOCK_AXIS)],
            x_label="average wallclock for this game (s)",
            y_label="average score for this game",
            title=f"{g} — score vs wallclock",
            figsize=figsize,
            overlay_curves=per_pass_wall or None,
        )
        html = (
            '<div class="per-game-card">'
            f'<div class="per-game-header"><span class="game-name">{thumb}<code>{escape(g)}</code></span></div>'
            f'<img src="data:image/png;base64,{tokens_b64}" alt="{escape(g)} tokens">'
            f'<img src="data:image/png;base64,{wall_b64}" alt="{escape(g)} wallclock">'
            "</div>"
        )
        cards.append(html)
        new_cache[g] = (key, html)
    cache_write = _per_game_card_cache_write(benchmark.job_dir, new_cache)
    if cache_write is not None:
        writes.append(cache_write)
    return '<div class="per-game-grid">' + "".join(cards) + "</div>"


def _per_game_grid_html(
    benchmarks: list[taaf.benchmark.Benchmark],
    common_games: list[str],
    labels: list[str],
) -> str:
    """Per-game cards on the comparison index. Each card holds the
    initial-state thumbnail + game id + two small clickable plots
    (tokens, wallclock) wrapped in a link to ``per_game/<id>.html`` —
    the same two plots again, but per game and clickable through.
    """
    cards: list[str] = []
    for g in common_games:
        g_runs = _game_runs_by_id(benchmarks[0], g)
        thumb = _thumbnail_html(g_runs[0], width_px=THUMB_FINAL_PX) if g_runs else ""
        tokens_b64, wall_b64 = _per_game_plots_b64(benchmarks, g, labels)
        href = f"per_game/{_safe(g)}.html"
        cards.append(
            '<div class="per-game-card">'
            f'<div class="per-game-header"><a href="{href}" class="game-link">{thumb}<code>{escape(g)}</code></a></div>'
            f'<a href="{href}"><img src="data:image/png;base64,{tokens_b64}" alt="{escape(g)} tokens"></a>'
            f'<a href="{href}"><img src="data:image/png;base64,{wall_b64}" alt="{escape(g)} wallclock"></a>'
            "</div>"
        )
    return '<h2>Per-game</h2><div class="per-game-grid">' + "".join(cards) + "</div>"


def _write_per_game_page(game_id: str, benchmarks: list[taaf.benchmark.Benchmark], out_path: Path) -> None:
    """Per-game drill-down page in the comparison tree.

    Same column shape as the per-run table, just sliced by game instead of
    by pass. Answer cells link across to each run's `<job_dir>/answers/...`
    via a path relative to this per-game page.
    """
    rows = [
        "<tr><th>run</th><th>pass</th><th>state</th><th>score</th>"
        "<th>correct</th><th>tokens</th><th>answers</th><th>note</th></tr>"
    ]
    representative_run: taaf.game.GameRun | None = None
    page_dir = out_path.parent  # so we can resolve answer-sheet paths relative to here
    for bm in benchmarks:
        for p in range(bm.n_passes):
            for r in _pass_runs(bm, p):
                if r.game_id != game_id:
                    continue
                if representative_run is None:
                    representative_run = r
                score = _arc_score(r)
                tokens = total_tokens(r)
                # Answer-sheet link — only if the benchmark wrote a job_dir AND the
                # sheet actually exists on disk, at
                # `answers/g{game_id}_p{pass}.html` inside the run's job_dir.
                answers_cell = "—"
                if bm.job_dir is not None:
                    stem = f"g{_safe(r.game_id)}_p{p}"
                    sheet_path = bm.job_dir / "answers" / f"{stem}.html"
                    if sheet_path.exists():
                        rel = os.path.relpath(sheet_path, start=page_dir)
                        thumb = _thumbnail_html(r, width_px=THUMB_FINAL_PX)
                        answers_cell = f'<a href="{rel}">{thumb or "view"}</a>'
                note_html = escape(r.solver_note) if r.solver_note else ""
                rows.append(
                    f"<tr><td><code>{escape(bm.label)}</code></td>"
                    f"<td>{p}</td>"
                    f"{_state_cell_html(r.state)}"
                    f"<td>{score:.2f}</td>"
                    f"<td>{_correct_cell(r)}</td>"
                    f"<td>{tokens}</td>"
                    f"<td>{answers_cell}</td>"
                    f"<td>{note_html}</td></tr>"
                )
    table_html = "<table>" + "".join(rows) + "</table>"
    thumb = _thumbnail_html(representative_run, width_px=128)
    # Full-size per-game plots above the table — same data as the small
    # ones on the comparison index, just larger.
    labels = [b.label or f"run {i}" for i, b in enumerate(benchmarks)]
    tokens_b64, wall_b64 = _per_game_plots_b64(benchmarks, game_id, labels, figsize=(8, 4))
    body = (
        '<p><a href="../index.html">← back to comparison</a></p>'
        f'<h1><span class="game-name">{thumb}<code>{escape(game_id)}</code></span></h1>'
        '<p class="intro">All runs of this single game across the benchmarks being compared. '
        "Plots show how each run progressed for THIS game in particular; the table at the "
        "bottom lists every (run, pass) cell with score and cost.</p>"
        "<h2>Score vs cumulative tokens</h2>"
        f'<img src="data:image/png;base64,{tokens_b64}" alt="{escape(game_id)} tokens">'
        "<h2>Score vs wallclock</h2>"
        f"{_PLOT_DESIGN_NOTE}"
        f'<img src="data:image/png;base64,{wall_b64}" alt="{escape(game_id)} wallclock">'
        "<h2>All (run, pass) cells</h2>"
        f"{table_html}"
    )
    out_path.write_text(_html_page(f"Comparison — {game_id}", body))


# --- Public: regenerate-from-disk helpers -----------------------------------


def _load_benchmark_from_dir(run_dir: Path) -> taaf.benchmark.Benchmark:
    """Load a finished benchmark from its ``job_dir``.

    Prefers ``benchmark.json`` (and the ``game_specs.pkl``
    sidecar that :meth:`Benchmark.from_json` attaches by default): that
    pair round-trips everything the diagnostics renderers (answer sheets,
    curves) need, without requiring
    the original solver / game classes to be importable. Falls back to
    the legacy monolithic ``benchmark.pkl`` for runs that predate the
    split layout.
    """
    json_path = run_dir / "benchmark.json"
    pkl_path = run_dir / "benchmark.pkl"
    if json_path.exists():
        # ``from_json`` auto-attaches game specs from its sidecar
        # and sets job_dir to the JSON's parent; no further wiring needed.
        return taaf.benchmark.Benchmark.from_json(json_path)
    if pkl_path.exists():
        bm = taaf.benchmark.Benchmark.from_pickle(pkl_path)
        bm.job_dir = run_dir
        return bm
    raise FileNotFoundError(f"No benchmark.json and no benchmark.pkl in {run_dir}")


def regenerate_run_diagnostics(
    benchmark: str | Path | taaf.benchmark.Benchmark,
    out_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Regenerate per-run diagnostics for a finished benchmark.
    ``benchmark`` is a run dir path or a ``Benchmark`` instance. When
    a path is passed, ``out_dir`` defaults to it (in-place regen).
    When a ``Benchmark`` instance is passed, ``out_dir`` is mandatory.

    Refuses in-flight inputs. If ``out_dir`` already holds a different
    ``benchmark.json``, refuses unless ``overwrite=True``. When the
    output dir is fresh, writes ``benchmark.json`` +
    ``game_specs.pkl`` there so it's self-contained for future
    regens. ``games.pkl`` / ``solver.pkl`` are not re-written.
    """
    if isinstance(benchmark, (str, Path)):
        input_dir = Path(benchmark)
        bm = _load_benchmark_from_dir(input_dir)
        if out_dir is None:
            out_dir = input_dir
    else:
        bm = benchmark
        if out_dir is None:
            raise ValueError("out_dir is required when passing a Benchmark instance (no implicit path to default to)")
    out_dir = Path(out_dir)

    if _is_in_flight(bm):
        raise ValueError(
            f"benchmark is in-flight (start_time={bm.start_time}, end_time=None). "
            f"regenerate_run_diagnostics only operates on finished runs; "
            f"re-run after teardown."
        )

    # JSON equality check against any existing snapshot at out_dir.
    # ``to_json_dict`` captures all diagnostically-relevant state and
    # dodges Python class-identity quirks.
    out_json = out_dir / "benchmark.json"
    must_write = True
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        if bm.to_json_dict() == existing:
            must_write = False
        elif not overwrite:
            raise ValueError(
                f"{out_json} already exists and does not match the provided "
                f"benchmark. Pass overwrite=True to replace it, choose a "
                f"different out_dir, or delete the existing snapshot first."
            )

    # Temporarily point ``job_dir`` at out_dir so the card cache and
    # save methods write there. Restored on exit.
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_job_dir = bm.job_dir
    bm.job_dir = out_dir
    try:
        if must_write:
            bm._save_json()
            bm._save_game_specs()
        generate_run_html(bm, out_dir / "diagnostics.html")
        generate_run_summary_txt(bm, out_dir / "summary.txt")
    finally:
        bm.job_dir = saved_job_dir


def regenerate_comparison_diagnostics(
    benchmarks: list[str | Path | taaf.benchmark.Benchmark],
    out_dir: str | Path,
) -> None:
    """Write the comparison HTML tree to ``out_dir`` from a mix of run
    directories and/or in-memory ``Benchmark`` instances. Refuses
    in-flight inputs.

    Only HTML lands at ``out_dir`` — per-run artifacts stay in their
    input dirs and per-game answer links use ``os.path.relpath``, so
    the input dirs must stay reachable from ``out_dir`` for the links
    to work.
    """
    bms: list[taaf.benchmark.Benchmark] = []
    for b in benchmarks:
        if isinstance(b, (str, Path)):
            bms.append(_load_benchmark_from_dir(Path(b)))
        else:
            bms.append(b)
    generate_comparison_html(bms, Path(out_dir))
