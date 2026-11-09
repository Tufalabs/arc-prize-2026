"""Fit ``ullm.reasoning_register``'s data file from labelled runs.

Two corpora of reasoning text -- runs served with the recurrent state
blanked, runs served consistently -- and one question: which words tell
them apart? The answer is Monroe, Colaresi and Quinn's log-odds ratio with
an informative Dirichlet prior ("Fightin' Words", 2008), z-scored, which is
the standard instrument for "typical of A rather than B": a frequency ratio
hands the list to rare words and raw log-odds hands it to stopwords.

The candidate vocabulary is a fixed pool -- pronouns, determiners,
auxiliaries, conjunctions, prepositions, discourse markers and a few verbs
of thought. Board colours and game names never enter, so the weights
read how the model writes rather than which game it is playing; they are
still fitted on the duck's reasoning and read that. The top words by |z|
are kept, each with its log-odds delta as the per-token weight.

Calibration is set rather than fitted: the classes are close to separable
at the game level, so a maximum-likelihood logistic fit is decided by a
handful of games and runs away from the rest. Instead the class means of
the per-game evidence map to 0.9 and 0.1, with the midpoint at 0.5.

Reads the runs through ``platypus.reasoning_corpus`` -- the one reader of
the span format's ``conversation`` and ``discarded_rounds`` artifacts --
which is why this lives in tools/ and not in ullm.

    uv run python tools/reasoning_register_data.py [--games N] [--vocab K] [--out PATH]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime
import json
import math
import pathlib
import statistics

from ullm.reasoning_register import DATA_PATH, OPENING_DATA_PATH, tokens

import platypus.reasoning_corpus

RESULTS = pathlib.Path("/data/shared/research/arc_3_results")

#: One run per class, each labelled by inspection rather than by how it was
#: served: the first stays in the telegraphic register for the whole game,
#: the second is verbose in every game. Whatever produced a register -- the
#: blank, a served opening, or the sampler on its own -- the classifier reads
#: the writing, so the cleanest corpus is the run that holds its register.
#: The served opening is stripped from the injected run's text before it is
#: counted (``strip_injected``), or the fit would learn the words we chose.
RUNS: dict[str, list[str]] = {
    "blanked": ["zr_inject_full_20260904/control_inject04w_later02w_4e5a6039"],
    "control": ["zr_sweep_20260902_181704/zr0_5f877e0"],
}

#: Words a register can be read from without reading the game. Anything
#: not in this pool is invisible to the fit.
CLOSED_CLASS: frozenset[str] = frozenset(
    """
    i me my mine myself we us our ours ourselves you your it its itself they them their
    this that these those the a an some any each every no none all both either neither
    is are was were be been being am do does did done have has had having
    can could may might must shall should will would ought
    and or but so because if then than although though while unless since whether
    of to in on at by for with from into over under about after before between through
    without within against during until upon
    not now just also maybe perhaps probably possibly actually okay ok hmm hm wait well
    first next again still yet only even very really quite rather already indeed
    alternatively however therefore thus hence anyway right sure fine good bad
    what which how why where when who whom whose
    let let's need needs try trying check checking see seeing look looking think thinking
    know knowing want seems seem means mean guess suppose assume consider test verify
    use using get got make made go going take give
    yes nope hmmm oh ah
    """.split()
)


def extract(per_run: int, opening: bool = False) -> dict[str, list[tuple[str, list[str]]]]:
    """scenario -> [(game label, its reasoning texts)], ``per_run`` games from each run.

    ``opening`` keeps only each game's FIRST reasoning text. The registers are
    furthest apart there and a caller can read it for one block, but the
    vocabulary that separates them is not the same one -- an opening's "let's"
    belongs to the CONTROL register ("Let's start by checking the ...") and a
    corpus's to the blanked one -- so the two are fitted separately.
    """
    jobs: list[tuple[str, str, str]] = []
    for scenario, runs in RUNS.items():
        for run_rel in runs:
            games = sorted(p.name for p in (RESULTS / run_rel / "games").iterdir() if p.is_dir())
            jobs += [(scenario, run_rel, g) for g in games[:per_run]]
    print(f"extracting {len(jobs)} games", flush=True)
    out: dict[str, list[tuple[str, list[str]]]] = {"blanked": [], "control": []}
    reader = _game_opening if opening else _game_texts
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        texts_by_job = pool.map(reader, [(r, g) for _, r, g in jobs])
        for (scenario, run_rel, game), texts in zip(jobs, texts_by_job):
            out[scenario].append((f"{run_rel.rsplit('/', 1)[-1]}/{game}", texts))
    return out


def _game_opening(job: tuple[str, str]) -> list[str]:
    """One game's opening reasoning text, read without walking the whole game."""
    run_rel, game = job
    text = platypus.reasoning_corpus.first_reasoning(RESULTS / run_rel / "games" / game, strip_injected=True)
    return [text] if text else []


def _game_texts(job: tuple[str, str]) -> list[str]:
    """One game's reasoning, in a worker process: block reads dominate, and a game is tens of seconds."""
    run_rel, game = job
    return platypus.reasoning_corpus.game_reasoning(RESULTS / run_rel / "games" / game, strip_injected=True)


def fightin_words(
    a: collections.Counter[str], b: collections.Counter[str], *, prior: float = 500.0
) -> list[tuple[float, float, str]]:
    """(z, delta, word) per candidate, most blanked first.

    ``delta`` is the log-odds difference under the informative Dirichlet
    prior -- the per-token weight -- and ``z`` is it over its standard error,
    which is what ranks the vocabulary.
    """
    pooled = a + b
    total = sum(pooled.values())
    n_a, n_b = sum(a.values()), sum(b.values())
    rows: list[tuple[float, float, str]] = []
    for word in CLOSED_CLASS:
        if pooled[word] == 0:
            continue
        alpha = prior * pooled[word] / total
        ya, yb = a[word] + alpha, b[word] + alpha
        delta = math.log(ya / (n_a + prior - ya)) - math.log(yb / (n_b + prior - yb))
        z = delta / math.sqrt(1.0 / ya + 1.0 / yb)
        rows.append((z, delta, word))
    return sorted(rows, reverse=True)


def evidence(weights: dict[str, float], texts: list[str]) -> float:
    """Mean per-token weight over all tokens, the statistic the classifier scores."""
    words = [w for t in texts for w in tokens(t)]
    return sum(weights.get(w, 0.0) for w in words) / len(words) if words else 0.0


def main() -> None:
    """Extract, fit, report, and write the data file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games read from each run")
    ap.add_argument("--vocab", type=int, default=30, help="words kept, by |z|")
    ap.add_argument("--min-count", type=int, default=50, help="pooled occurrences a word needs to be kept")
    ap.add_argument(
        "--slice",
        choices=("corpus", "opening"),
        default="corpus",
        help="fit on whole games, or on their opening reasoning text alone",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None, help="defaults to the data file for --slice")
    args = ap.parse_args()
    opening = args.slice == "opening"
    out_path = args.out or (OPENING_DATA_PATH if opening else DATA_PATH)

    corpus = extract(args.games, opening=opening)
    counts = {
        scenario: collections.Counter(w for _, texts in games for t in texts for w in tokens(t))
        for scenario, games in corpus.items()
    }
    ranked = fightin_words(counts["blanked"], counts["control"])
    pooled = counts["blanked"] + counts["control"]
    kept = sorted((r for r in ranked if pooled[r[2]] >= args.min_count), key=lambda r: abs(r[0]), reverse=True)[
        : args.vocab
    ]
    weights = {word: delta for _, delta, word in kept}

    # Per-game evidence: one number per game, the unit the calibration is set on.
    per_game = {
        scenario: [evidence(weights, texts) for _, texts in games if texts] for scenario, games in corpus.items()
    }
    mu = {s: statistics.mean(v) for s, v in per_game.items()}
    sd = {s: statistics.stdev(v) for s, v in per_game.items()}
    # Class means to 0.9 and 0.1, midpoint to 0.5: logit(0.9) = ln 9.
    scale = 2.0 * math.log(9.0) / (mu["blanked"] - mu["control"])
    offset = -scale * (mu["blanked"] + mu["control"]) / 2.0

    print(f"\n{'word':<14} {'z':>7} {'delta':>7} {'blanked':>9} {'control':>9}")
    for z, delta, word in sorted(kept, key=lambda r: r[0], reverse=True):
        print(f"{word:<14} {z:>7.1f} {delta:>7.2f} {counts['blanked'][word]:>9,} {counts['control'][word]:>9,}")
    print(f"\nper-game evidence: blanked {mu['blanked']:.4f} +- {sd['blanked']:.4f} (n={len(per_game['blanked'])}),")
    print(f"                   control {mu['control']:.4f} +- {sd['control']:.4f} (n={len(per_game['control'])})")
    worst_blanked, best_control = min(per_game["blanked"]), max(per_game["control"])
    print(f"separation: lowest blanked game {worst_blanked:.4f}, highest control game {best_control:.4f}")

    data = {
        "weights": dict(sorted(weights.items())),
        "calibration": {"scale": scale, "offset": offset},
        "provenance": {
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "generator": "tools/reasoning_register_data.py",
            "runs": RUNS,
            "games_per_run": args.games,
            "texts": {s: sum(len(t) for _, t in g) for s, g in corpus.items()},
            "tokens": {s: sum(c.values()) for s, c in counts.items()},
            "vocabulary_z": {word: z for z, _, word in kept},
            "per_game_evidence_mean": mu,
            "per_game_evidence_sd": sd,
        },
    }
    out_path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
