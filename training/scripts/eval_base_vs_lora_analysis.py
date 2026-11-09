"""Paired analysis of a base-vs-LoRA eval run (scripts/eval_base_vs_lora.py).

Usage::

    python3 scripts/eval_base_vs_lora_analysis.py <eval_dir> \
        [--base-budget 80000] [--lora-budget 80000] \
        [--grid-step 2000] [--bootstrap 10000] [--bootstrap-seed 20260830]

Reads ``<eval_dir>/episodes_base.jsonl`` and ``episodes_lora.jsonl`` and
computes, in GENERATED (completion) tokens:

- per-(game, seed) paired scores at the arm token budgets
  (80k generated tokens for BOTH arms by default),
- per-game mean paired diff (lora - base),
- the overall mean diff with a bootstrap CI over the episode pairs,
- win/tie/loss counts,
- ``<eval_dir>/curves.json``: for a token-budget grid (0..base budget, step
  2k), the mean score across ALL ok episodes of each arm at that budget.

"Score at token budget B" walks each episode's per-step trace (cumulative
generated tokens, score after the step) and takes the score after the last
step whose cumulative generated tokens fit inside B. This post-hoc
truncation is exact: capping generation cannot change the history before
the cap. A step that STRADDLES B contributes nothing (its completion would
not have finished within the budget, so the actions it authorized never
happen). An episode that ended before B — finished the game, gave up, or
hit its runtime cap — contributes its final score at every larger budget:
its trace simply has no steps beyond its end, so the walk returns the last
recorded score. An arm's curve is therefore flat beyond the tokens its
episodes actually produced (e.g. when the 60-minute wall-clock backstop
ended an episode before 80k generated tokens); that is the honest reading,
not an artifact. Scores only move on level completion (the ARC score has
no finer granularity), so per-episode curves are step functions.

Stdlib only — runs anywhere, no venv needed. Plotting happens elsewhere;
this only prints the summary table and emits curves.json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_TIE_EPS = 1e-9


def _load_episodes(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Return ({(game, seed): ok_record}, [non-ok records]).

    A later ok line for the same (game, seed) wins — the driver appends a
    fresh line when a failed episode is retried on resume.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    ok: dict[tuple[str, int], dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(record["game"]), int(record["seed"]))
            if record.get("status") == "ok":
                ok[key] = record
            else:
                bad.append(record)
    # A key that failed first and succeeded on retry is not "bad".
    bad = [r for r in bad if (str(r["game"]), int(r["seed"])) not in ok]
    return ok, bad


def score_at_budget(record: dict[str, Any], budget: int) -> float:
    """Score after the last step whose cumulative generated tokens <= budget."""
    score = 0.0
    for step in record.get("steps") or []:
        if int(step["cum_output_tokens"]) > budget:
            break
        score = float(step["score_after"])
    return score


def _bootstrap_ci(
    diffs: list[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        means.append(sum(rng.choice(diffs) for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir")
    parser.add_argument("--base-budget", type=int, default=80_000)
    parser.add_argument("--lora-budget", type=int, default=80_000)
    parser.add_argument("--grid-step", type=int, default=2_000)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    base_ok, base_bad = _load_episodes(eval_dir / "episodes_base.jsonl")
    lora_ok, lora_bad = _load_episodes(eval_dir / "episodes_lora.jsonl")

    for arm, bad in (("base", base_bad), ("lora", lora_bad)):
        for record in bad:
            print(
                f"WARNING: unresolved failed episode [{arm}] game={record.get('game')} "
                f"seed={record.get('seed')} status={record.get('status')} "
                f"error={str(record.get('error', ''))[:120]}"
            )

    paired_keys = sorted(set(base_ok) & set(lora_ok))
    missing = sorted((set(base_ok) | set(lora_ok)) - set(paired_keys))
    for key in missing:
        side = "lora" if key in base_ok else "base"
        print(
            f"WARNING: unpaired episode {key[0]} seed={key[1]} (missing on {side} arm)"
        )
    if not paired_keys:
        print("ERROR: no paired (game, seed) episodes; nothing to analyze")
        return 1

    # --- (a) paired scores at the arm budgets -----------------------------
    pairs: list[tuple[tuple[str, int], float, float]] = []
    for key in paired_keys:
        base_score = score_at_budget(base_ok[key], args.base_budget)
        lora_score = score_at_budget(lora_ok[key], args.lora_budget)
        pairs.append((key, base_score, lora_score))

    # --- (b) per-game mean diff -------------------------------------------
    games = sorted({key[0] for key, _, _ in pairs})
    per_game: dict[str, dict[str, float]] = {}
    for game in games:
        rows = [(base, lora) for (g, _), base, lora in pairs if g == game]
        base_mean = statistics.fmean(b for b, _ in rows)
        lora_mean = statistics.fmean(lora for _, lora in rows)
        per_game[game] = {
            "n": len(rows),
            "base_mean": base_mean,
            "lora_mean": lora_mean,
            "diff_mean": lora_mean - base_mean,
        }

    # --- (c) overall mean diff + bootstrap CI over the pairs --------------
    diffs = [lora - base for _, base, lora in pairs]
    overall_base = statistics.fmean(b for _, b, _ in pairs)
    overall_lora = statistics.fmean(lora for _, _, lora in pairs)
    overall_diff = statistics.fmean(diffs)
    ci_lo, ci_hi = _bootstrap_ci(
        diffs, resamples=args.bootstrap, seed=args.bootstrap_seed
    )

    # --- (d) win/tie/loss --------------------------------------------------
    wins = sum(1 for d in diffs if d > _TIE_EPS)
    losses = sum(1 for d in diffs if d < -_TIE_EPS)
    ties = len(diffs) - wins - losses

    # --- (e) curves.json ---------------------------------------------------
    budgets = list(range(0, args.base_budget + 1, args.grid_step))
    base_records = list(base_ok.values())
    lora_records = list(lora_ok.values())
    curves = {
        "budget_grid_tokens": budgets,
        "arms": {
            "base": {
                "n_episodes": len(base_records),
                "eval_budget_tokens": args.base_budget,
                "mean_score": [
                    statistics.fmean(score_at_budget(r, b) for r in base_records)
                    for b in budgets
                ],
            },
            "lora": {
                "n_episodes": len(lora_records),
                "eval_budget_tokens": args.lora_budget,
                "mean_score": [
                    statistics.fmean(score_at_budget(r, b) for r in lora_records)
                    for b in budgets
                ],
            },
        },
        "note": (
            "mean official ARC score across all ok episodes of each arm at "
            "each generated-token budget; an episode that ended (won / gave "
            "up / hit its runtime cap) before a budget contributes its final "
            "score there, so each arm's curve is flat beyond the tokens its "
            "runtime cap actually produced"
        ),
    }
    curves_path = eval_dir / "curves.json"
    curves_path.write_text(json.dumps(curves, indent=2), encoding="utf-8")

    # --- summary table ------------------------------------------------------
    print()
    print(
        f"paired eval: {len(pairs)} (game, seed) pairs over {len(games)} games "
        f"(base@{args.base_budget} vs lora@{args.lora_budget} generated tokens)"
    )
    print()
    header = f"{'game':<16} {'n':>3} {'base':>8} {'lora':>8} {'diff':>8}"
    print(header)
    print("-" * len(header))
    for game in games:
        row = per_game[game]
        print(
            f"{game:<16} {row['n']:>3} {row['base_mean']:>8.3f} "
            f"{row['lora_mean']:>8.3f} {row['diff_mean']:>+8.3f}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<16} {len(pairs):>3} {overall_base:>8.3f} "
        f"{overall_lora:>8.3f} {overall_diff:>+8.3f}"
    )
    print()
    print(
        f"mean diff (lora - base): {overall_diff:+.3f}  "
        f"[95% bootstrap CI {ci_lo:+.3f} .. {ci_hi:+.3f}, "
        f"{args.bootstrap} resamples]"
    )
    print(f"win/tie/loss (lora vs base, per pair): {wins}/{ties}/{losses}")
    if missing or base_bad or lora_bad:
        print(
            f"NOTE: {len(missing)} unpaired and "
            f"{len(base_bad) + len(lora_bad)} failed episodes were excluded "
            "from the paired stats (see warnings above)"
        )
    print(f"curves written to {curves_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
