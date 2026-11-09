"""What each level of a seal run cost: completion tokens to beat it, and where games stopped.

A level's tokens are the completion tokens of the actions played on it, from
``benchmark.json``'s per-action history (each action carries what was generated
since the one before), plus the game's trailing tokens for the level it ended
on. That is the count the per-level budget charges, so a level that ran into
the cap sums to the cap exactly.

A level a game reached but did not beat is censored, not a cost: it says only
that beating it would have taken more than it spent. Those are kept apart by
why the game stopped there -- the per-level cap, the per-game clock, a crash,
or still playing (an in-flight run).

Levels print 1-based (L1 is the first), as the games number them.

    uv run python packages/seal/tools/level_tokens.py <run_dir> [<run_dir> ...] [--csv out.csv]

Several run dirs of the same games pool their passes.
"""

import argparse
import collections
import csv
import json
import os
import statistics
from typing import Any

BIN = 25_000

#: One level a pass reached, as ``attempts`` records it.
Row = dict[str, Any]


def _end_reason(game: dict[str, Any]) -> str:
    """Why a game that did not win stopped on the level it stopped on."""
    note = game.get("solver_note") or ""
    if game["state"] == "won":
        return "won"
    if "per-level token budget" in note:
        return "cap"
    if "runtime limit" in note:
        return "clock"
    if note.startswith("agent error") or game["state"] == "crashed":
        return "crash"
    if game["state"] == "playing":
        return "playing"
    return f"other:{game['state']}"


def attempts(run_dir: str) -> list[Row]:
    """One record per (game, pass, level reached)."""
    with open(os.path.join(run_dir, "benchmark.json")) as f:
        bench = json.load(f)
    out: list[Row] = []
    for game in bench["game_runs"]:
        history, per_level = game["history"], game["actions_per_level"]
        if sum(per_level) != len(history):
            raise ValueError(f"{game['game_id']} pass {game['pass_index']}: actions_per_level does not cover history")
        reason = _end_reason(game)
        start, wall_before = 0, 0.0
        for level, n_actions in enumerate(per_level):
            beaten = level < game["levels_completed"]
            last_reached = level == game["levels_completed"]
            if not beaten and not last_reached:
                break
            actions = history[start : start + n_actions]
            generated = sum(a["generated_tokens"] for a in actions)
            uncached = sum(a["uncached_input_tokens"] for a in actions)
            wall_after = actions[-1]["wallclock_seconds"] if actions else wall_before
            if last_reached:
                generated += game["final_generated_tokens"]
                uncached += game["final_uncached_input_tokens"]
                wall_after = max(wall_after, game["final_wallclock_seconds"])
            out.append(
                {
                    "run": run_dir,
                    "game": game["game_id"],
                    "pass": game["pass_index"],
                    "level": level,
                    "n_levels": game["number_of_levels"],
                    "beaten": beaten,
                    "outcome": "beaten" if beaten else reason,
                    "generated": generated,
                    "uncached_input": uncached,
                    "actions": n_actions,
                    "baseline_actions": game["base_actions_per_level"][level],
                    "wall_s": wall_after - wall_before,
                }
            )
            start += n_actions
            wall_before = wall_after
    return out


def _k(tokens: float) -> str:
    """Tokens in thousands."""
    return f"{tokens / 1000:.0f}k"


def report(rows: list[Row], cap: int | None) -> None:
    """Print the pooled distribution, the survival by token bin, and the per-game table."""
    beaten = [r for r in rows if r["beaten"]]
    censored = [r for r in rows if not r["beaten"]]
    games = sorted({r["game"] for r in rows})
    n_runs = len({(r["run"], r["game"], r["pass"]) for r in rows})
    print(f"{n_runs} game runs over {len(games)} games; {len(rows)} levels reached, {len(beaten)} beaten")
    print("unbeaten, by why the game stopped there:", dict(collections.Counter(r["outcome"] for r in censored)))

    tokens = sorted(r["generated"] for r in beaten)
    q = statistics.quantiles(tokens, n=20)
    print(
        f"\ntokens to beat a level (n={len(tokens)}): median {_k(statistics.median(tokens))}, "
        f"p25 {_k(q[4])}, p75 {_k(q[14])}, p90 {_k(q[17])}, p95 {_k(q[18])}, max {_k(tokens[-1])}"
    )

    # Survival over the spend: of the level attempts still unbeaten when a bin
    # starts, how many are beaten inside it. Censored attempts leave the risk
    # set at their own spend, so a cap-stop counts as alive through the cap.
    top = cap or max(r["generated"] for r in rows)
    print(f"\nper {_k(BIN)} bin: attempts alive at its start, beaten inside it, and the rate")
    print(f"  {'bin':>11} {'alive':>6} {'beaten':>7} {'rate':>6} {'cum beaten':>11}")
    lo, cum = 0, 0
    while lo < top:
        hi = lo + BIN
        alive = rows if lo == 0 else [r for r in rows if r["generated"] > lo]
        won = [r for r in alive if r["beaten"] and r["generated"] <= hi]
        cum += len(won)
        rate = len(won) / len(alive) if alive else 0.0
        print(f"  {_k(lo):>5}-{_k(hi):<5} {len(alive):>6} {len(won):>7} {rate:>6.0%} {cum:>11}")
        lo = hi

    print("\nper game: each pass's stopping level (W = won all), then per level beaten/reached and tokens to beat")
    by_game: dict[str, list[Row]] = collections.defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    for game in games:
        g_rows = by_game[game]
        n_levels = g_rows[0]["n_levels"]
        passes = sorted({(r["run"], r["pass"]) for r in g_rows})
        stops: list[str] = []
        for p in passes:
            p_rows = [r for r in g_rows if (r["run"], r["pass"]) == p]
            last = max(p_rows, key=lambda r: r["level"])
            stops.append(
                "W"
                if last["beaten"]
                else f"L{last['level'] + 1}{'' if last['outcome'] == 'cap' else '(' + last['outcome'] + ')'}"
            )
        died: collections.Counter[int] = collections.Counter(
            r["level"] for r in g_rows if not r["beaten"] and r["outcome"] == "cap"
        )
        killer = died.most_common(1)[0] if died else None
        print(
            f"\n{game}  ({n_levels} levels)  stops: {' '.join(stops)}"
            + (f"  killer: L{killer[0] + 1} ({killer[1]}/{len(passes)})" if killer else "")
        )
        for level in range(n_levels):
            at = [r for r in g_rows if r["level"] == level]
            if not at:
                continue
            won = sorted(r["generated"] for r in at if r["beaten"])
            lost = collections.Counter(r["outcome"] for r in at if not r["beaten"])
            cost = " ".join(_k(t) for t in won)
            lost_s = " ".join(f"{n}x{o}" for o, n in lost.items())
            print(
                f"   L{level + 1:<2} {len(won)}/{len(at)} beaten  [{cost}]"
                + (f"  unbeaten: {lost_s}" if lost_s else "")
            )

    # A killer is a level that stops at least as many passes as it lets
    # through. A level only one pass reached says too little to call.
    print("\nkiller levels: reached by 2+ passes and beaten by at most half of them")
    print(f"  {'game':<16} {'level':<5} {'beaten':>7} {'cap-stops':>9}  tokens of the passes that beat it")
    table: list[tuple[float, int, str, int, list[int], int, int]] = []
    for (game, level), at in _by_level(rows).items():
        won = sorted(r["generated"] for r in at if r["beaten"])
        caps = sum(r["outcome"] == "cap" for r in at)
        if len(at) >= 2 and caps and len(won) * 2 <= len(at):
            table.append((len(won) / len(at), -caps, game, level, won, len(at), caps))
    for _, _, game, level, won, n_at, caps in sorted(table):
        print(f"  {game:<16} L{level + 1:<4} {len(won):>3}/{n_at:<3} {caps:>9}  {' '.join(_k(t) for t in won)}")

    # What a retry could reach: a pass that ran into the cap on a level some
    # other pass of the same game beat. The expectation takes, for each
    # cap-stop, the share of the level's other attempts that beat it.
    by_level = _by_level(rows)
    cap_stops = [r for r in censored if r["outcome"] == "cap"]
    beaten_elsewhere = [r for r in cap_stops if any(o["beaten"] for o in by_level[(r["game"], r["level"])])]
    expected = sum(
        sum(o["beaten"] for o in others) / len(others)
        for r in cap_stops
        if (others := [o for o in by_level[(r["game"], r["level"])] if o is not r])
    )
    print(
        f"\ncap-stops on a level another pass beat: {len(beaten_elsewhere)} of {len(cap_stops)}; "
        f"a second full-budget try, if it did as the level's other attempts did, beats {expected:.0f} of them"
    )


def _by_level(rows: list[Row]) -> dict[tuple[str, int], list[Row]]:
    """Level attempts grouped by (game, level)."""
    grouped: dict[tuple[str, int], list[Row]] = collections.defaultdict(list)
    for r in rows:
        grouped[(r["game"], r["level"])].append(r)
    return grouped


def main() -> None:
    """Load the runs, print the report, optionally write the per-level rows."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--cap", type=int, default=None, help="the per-level cap, to end the survival table on it")
    parser.add_argument("--csv", help="write one row per level reached")
    args = parser.parse_args()
    rows = [r for d in args.run_dirs for r in attempts(d)]
    report(rows, args.cap)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
