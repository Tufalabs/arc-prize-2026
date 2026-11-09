"""Build expert-pruning plans from replayed routing statistics (CPU).

Usage:
    python scripts/prune_plans.py --stats <prune_route_stats out> --out <plans dir> \\
        --holdout <regex on the rollout source> [--keep 307 256] [--beta 1.0] \\
        [--reference <pruned checkpoint dir with pruning.json> ...]

Calibration rollouts are every rollout whose source path does not match
``--holdout`` (all of them without it); held-out ones are kept for the probe. Criteria, each a
``{"num_experts", "kept"}`` plan for ``prune_flashnext_experts.py``:

- ``reap``: REAP's mean of gate * output norm over the tokens routed to an
  expert (arXiv 2510.13999), all calibration rollouts alike;
- ``mass``: the same contribution per replayed token, frequency included;
- ``mass_scored``: mass with each rollout weighted ``exp(beta * z)``, z its
  score standardised among the attempts at the same game or puzzle;
- ``reap_enriched``: REAP times the expert's enrichment in high-scoring
  rollouts (``mass_scored / mass``);
- ``reap_scored``: REAP's mean over the routed tokens with each rollout's
  tokens weighted ``exp(beta * z)``, so high-scoring rollouts set it.

``report.json`` records the calibration split and each plan's overlap with the
others and with ``--reference`` plans (e.g. e307's ``pruning.json``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.flash_route_stats import keep_plan, mass_scores, reap_scores, within_game_weights  # noqa: E402


def load_stats(root: Path) -> tuple[dict, dict]:
    """Per-run arrays and metadata for every replayed rollout."""
    arrays, meta = {}, {}
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("verify_"):
            continue
        record = json.loads(path.read_text())
        npz = path.with_suffix(".npz")
        if not npz.exists():
            continue
        data = np.load(npz)
        arrays[record["run_id"]] = {
            "sum_gnorm": data["sum_gnorm"],
            "sum_g": data["sum_g"],
            "count": data["count"],
            "tokens": record["tokens"],
        }
        meta[record["run_id"]] = record
    return arrays, meta


def provenance(meta: dict[str, dict]) -> dict:
    """What the calibration rollouts were replayed through and from: the
    checkpoint, whether their images went through the vision tower, and the
    recordings. A plan carries it into the pruned checkpoint's ``pruning.json``,
    which otherwise names neither. Statistics from before a field was recorded
    give None for it; rollouts that disagree are not one calibration."""
    out: dict = {}
    for field, key in (("source_checkpoint", "checkpoint"), ("replayed_images", "multimodal")):
        values = {record.get(key) for record in meta.values()}
        if len(values) > 1:
            raise SystemExit(f"calibration rollouts disagree on {key}: {sorted(map(str, values))}")
        out[field] = values.pop() if values else None
    out["calibration_recordings"] = sorted(
        {str(Path(record["steps"]).parents[2]) for record in meta.values() if record.get("steps")}
    )
    return out


def overlap(first: dict, second: dict) -> float:
    """Mean per-layer fraction of kept experts two plans share."""
    layers = sorted(first["kept"], key=int)
    return float(
        np.mean([len(set(first["kept"][k]) & set(second["kept"][k])) / len(first["kept"][k]) for k in layers])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--holdout", default=None, help="regex matched against each rollout's source path (none: calibrate on all)"
    )
    parser.add_argument("--keep", type=int, nargs="+", default=[307, 256])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--reference", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    arrays, meta = load_stats(args.stats)
    holdout = re.compile(args.holdout) if args.holdout else None
    calibration = {run: arrays[run] for run in arrays if not (holdout and holdout.search(meta[run]["source"]))}
    held_out = sorted(run for run in arrays if holdout and holdout.search(meta[run]["source"]))
    if not calibration or (holdout and not held_out):
        raise SystemExit(f"need calibration (and, with --holdout, held-out) rollouts: {len(calibration)} / {len(held_out)}")
    scores = {run: float(meta[run]["score"]) for run in calibration}
    games = {run: meta[run]["game_id"] for run in calibration}
    weights = within_game_weights(scores, games, beta=args.beta)

    reap = reap_scores(calibration)
    mass = mass_scores(calibration)
    scored = mass_scores(calibration, weights)
    enrichment = np.divide(scored, mass, out=np.ones_like(scored), where=mass > 0)
    criteria = {
        "reap": reap,
        "mass": mass,
        "mass_scored": scored,
        "reap_enriched": reap * enrichment,
        "reap_scored": reap_scores(calibration, weights),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    replayed = provenance({run: meta[run] for run in calibration})
    plans = {}
    for keep in args.keep:
        for name, values in criteria.items():
            label = f"{name}_k{keep}"
            plan = keep_plan(
                values,
                keep=keep,
                source={
                    "criterion": name,
                    "beta": args.beta if "scored" in name or "enriched" in name else None,
                    "calibration_rollouts": len(calibration),
                    "stats": str(args.stats),
                    **replayed,
                },
            )
            (args.out / f"{label}.json").write_text(json.dumps(plan, indent=1))
            plans[label] = plan
    references = {}
    for path in args.reference:
        pruning = json.loads((path / "pruning.json").read_text())
        references[path.name] = {"num_experts": pruning["num_experts"], "kept": pruning["kept"]}
    report = {
        "calibration": sorted(calibration),
        "held_out": held_out,
        "calibration_games": len(set(games.values())),
        "weights_spread": [min(weights.values()), max(weights.values())],
        "overlap": {
            label: {
                **{
                    other: overlap(plan, plans[other])
                    for other in plans
                    if other != label and other.rsplit("_k", 1)[1] == label.rsplit("_k", 1)[1]
                },
                **{name: overlap(plan, ref) for name, ref in references.items() if len(ref["kept"]["0"]) == len(plan["kept"]["0"])},
            }
            for label, plan in plans.items()
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("calibration_games", "weights_spread")}), f"calibration={len(calibration)} held_out={len(held_out)}")
    for label, row in report["overlap"].items():
        print(label, {k: round(v, 3) for k, v in row.items()})


if __name__ == "__main__":
    main()
