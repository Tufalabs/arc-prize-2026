"""Rank a Flash-Next histogram's experts and write the plan a prune executes.

Input is a routing histogram ``ullm.expert_log`` wrote during a calibration
run: counts and gate-weight mass per (layer, expert), and with norms on, the
gate weight times the expert's output norm. Output is a plan naming the
experts every layer KEEPS, ascending, which ``prune_flashnext_experts.py``
turns into a checkpoint.

The criterion is a rule of three independent switches on the per-routing
term. The term is the gate weight; ``norm`` multiplies it by the L2 norm of
the expert's output, ``square`` squares it, and the terms are summed over
the routings that chose the expert. ``conditional`` divides that sum by the
count, turning total contribution into contribution per activation -- REAP's
criterion is norm + conditional. ``by="counts"`` is the plain frequency
criterion and takes none of the three. Every combination reads one
pre-summed histogram table: the sum of the squared term is recorded
separately, since a sum cannot be squared after the fact.

Allocation is uniform: the same number of experts leaves every layer, each
ranked within itself. The model code takes one expert count for all layers,
so a physical prune can do nothing else. No layer is left with fewer than
``top_k`` survivors, since routing needs that many to choose from.

The histogram is validated against the checkpoint before anything is
chosen: it must cover exactly the decoder layers the checkpoint's tensor
index gives a router, name no expert outside the checkpoint's count, and
each criterion needs a histogram that recorded its table. Applying a stale or mismatched histogram would prune
the wrong experts without a trace.

Usage: expert_prune_plan.py --histogram routed_experts.N.json --src SNAPSHOT --ratio 0.4 --out PLAN.json
       [--by mass|counts] [--norm] [--square] [--conditional]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PruneRule:
    """How a (layer, expert) pair is scored before ranking: three independent switches."""

    by: str = "mass"
    norm: bool = False
    square: bool = False
    conditional: bool = False

    def __post_init__(self) -> None:
        if self.by not in ("mass", "counts"):
            raise ValueError(f"by must be 'mass' or 'counts', got {self.by!r}")
        if self.by == "counts" and (self.norm or self.square or self.conditional):
            raise ValueError("by='counts' takes no norm/square/conditional switches")

    @property
    def table_key(self) -> str:
        """The histogram table holding the summed per-routing term."""
        if self.by == "counts":
            return "counts_by_layer"
        return {
            (False, False): "mass_by_layer",
            (False, True): "mass_sq_by_layer",
            (True, False): "gate_norm_by_layer",
            (True, True): "gate_norm_sq_by_layer",
        }[(self.norm, self.square)]

    def label(self) -> str:
        """Short name for logs and plan files, e.g. ``mass+norm+cond``."""
        switches = (("norm", self.norm), ("sq", self.square), ("cond", self.conditional))
        return "+".join([self.by] + [n for n, on in switches if on])


@dataclass
class PrunePlan:
    """Which experts every layer keeps, and what leaving the rest removes."""

    kept: dict[int, list[int]] = field(default_factory=lambda: dict[int, list[int]]())
    #: share of the layer's traffic (by the chosen criterion) the removed set carried
    removed_share: dict[int, float] = field(default_factory=lambda: dict[int, float]())
    #: layers where the top_k floor kept something the ratio would have removed
    floored: list[int] = field(default_factory=lambda: list[int]())

    def summary(self) -> str:
        """One line per layer; the log is how a plan is read before it is applied."""
        lines = [
            f"layer {lay}: kept {len(ids)} ({self.removed_share.get(lay, 0.0):.1%} of traffic removed)"
            + ("  [floored]" if lay in self.floored else "")
            for lay, ids in sorted(self.kept.items())
        ]
        total = sum(len(v) for v in self.kept.values())
        return f"{total} experts kept across {len(self.kept)} layers\n" + "\n".join(lines)


def criterion_rows(histogram: dict[str, Any], rule: PruneRule, layer_ids: list[int]) -> dict[int, dict[int, float]]:
    """The per-(layer, expert) values a plan ranks by, under ``rule``.

    Validated against the model's layers here so a stale or mismatched
    histogram is refused before anything is chosen. A conditional rule
    divides by counts; an expert with a value and no count is impossible in
    a histogram ``ullm.expert_log`` wrote, and reads as zero rather than
    infinite.
    """
    key = rule.table_key
    table: dict[str, dict[str, float]] = histogram.get(key) or {}
    if not table:
        hint = "an older histogram records counts only -- use by='counts'"
        if key.startswith("gate_norm"):
            hint = "norms are recorded only by a run with ULLM_EXPERT_NORMS set"
        elif key == "mass_sq_by_layer":
            hint = "squared gate weights are recorded only by histograms written after 2026-09-05"
        raise ValueError(f"histogram has no {key!r}; {hint}")
    hist_layers = sorted(int(k) for k in table)
    if hist_layers != sorted(layer_ids):
        raise ValueError(f"histogram layers {hist_layers} do not match the model's MoE layers {sorted(layer_ids)}")
    counts: dict[str, dict[str, float]] = histogram.get("counts_by_layer") or {}
    rows: dict[int, dict[int, float]] = {}
    for lay in layer_ids:
        row = {int(e): float(v) for e, v in table[str(lay)].items()}
        if rule.conditional:
            layer_counts = {int(e): float(c) for e, c in (counts.get(str(lay)) or {}).items()}
            row = {e: (v / layer_counts[e] if layer_counts.get(e) else 0.0) for e, v in row.items()}
        rows[lay] = row
    return rows


def plan_prune(
    histogram: dict[str, Any],
    *,
    ratio: float,
    rule: PruneRule,
    layer_ids: list[int],
    num_experts: int,
    top_k: int,
) -> PrunePlan:
    """Remove ``round(ratio * num_experts)`` experts from every layer, least-used first.

    Pairs are ranked by their within-layer share of the chosen criterion.
    No layer is left with fewer than ``top_k`` survivors.
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"prune ratio must be in [0, 1), got {ratio}")
    rows = criterion_rows(histogram, rule, layer_ids)
    per_layer = int(round(ratio * num_experts))
    plan = PrunePlan()
    for lay in layer_ids:
        row = rows[lay]
        bad = [e for e in row if not 0 <= e < num_experts]
        if bad:
            raise ValueError(f"histogram layer {lay} names expert ids {bad[:5]} outside 0..{num_experts - 1}")
        total = sum(row.values()) or 1.0
        ranked = sorted((row.get(e, 0.0) / total, e) for e in range(num_experts))
        n_remove = per_layer
        if num_experts - n_remove < top_k:
            # Give back the highest-share picks until routing has enough to
            # choose from. Ranking is by share, so this keeps the plan's
            # intent as closely as the floor allows.
            n_remove = num_experts - top_k
            plan.floored.append(lay)
        removed = ranked[:n_remove]
        plan.kept[lay] = sorted(e for _, e in ranked[n_remove:])
        plan.removed_share[lay] = sum(s for s, _ in removed)
    return plan


def model_facts(src: Path) -> tuple[int, int]:
    """``(num_experts, top_k)`` from the checkpoint's own config."""
    cfg = json.loads((src / "config.json").read_text())
    text = cfg.get("text_config") or cfg
    return int(text["num_experts"]), int(text["num_experts_per_tok"])


GATE_RE = re.compile(r"^.*\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
DECODER = "language_model.layers."


def moe_layer_ids(src: Path) -> list[int]:
    """The decoder layers with a router, from the checkpoint's own tensor index.

    The set the histogram must cover exactly; the MTP block's router is not
    a decoder layer and is left out, as the cutter leaves it untouched.
    """
    index = json.loads((src / "model.safetensors.index.json").read_text())
    ids = sorted({int(m["layer"]) for key in index["weight_map"] if (m := GATE_RE.match(key)) and DECODER in key})
    if not ids:
        raise ValueError(f"{src} indexes no decoder router tensor; not a checkpoint this planner knows")
    return ids


def main() -> None:
    """Read the histogram and the checkpoint's config, rank, write the plan."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--histogram", type=Path, required=True, help="routed_experts.<pid>.json from a calibration run")
    ap.add_argument("--src", type=Path, required=True, help="the checkpoint the histogram was measured on")
    ap.add_argument("--ratio", type=float, required=True, help="share of every layer's experts to remove")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--by", choices=("mass", "counts"), default="mass")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--conditional", action="store_true")
    args = ap.parse_args()

    rule = PruneRule(by=args.by, norm=args.norm, square=args.square, conditional=args.conditional)
    histogram: dict[str, Any] = json.loads(args.histogram.read_text())
    num_experts, top_k = model_facts(args.src)
    recorded = histogram.get("num_experts")
    if recorded is not None and int(recorded) != num_experts:
        raise ValueError(f"histogram was measured on {recorded} experts, the checkpoint has {num_experts}")
    # The checkpoint says which layers a plan must cover; a histogram that
    # covers another set is refused inside plan_prune rather than trusted.
    layer_ids = moe_layer_ids(args.src)
    plan = plan_prune(histogram, ratio=args.ratio, rule=rule, layer_ids=layer_ids, num_experts=num_experts, top_k=top_k)
    payload = {
        "kept": {str(lay): ids for lay, ids in sorted(plan.kept.items())},
        "num_experts": num_experts,
        "num_kept": len(next(iter(plan.kept.values()))) if plan.kept else num_experts,
        "ratio": args.ratio,
        "rule": rule.label(),
        "histogram": str(args.histogram),
        "histogram_sha256": hashlib.sha256(args.histogram.read_bytes()).hexdigest(),
        "src": str(args.src),
        "removed_share": {str(lay): share for lay, share in sorted(plan.removed_share.items())},
        "floored": plan.floored,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"rule {rule.label()}, ratio {args.ratio}\n{plan.summary()}\nwrote {args.out}")


if __name__ == "__main__":
    main()
