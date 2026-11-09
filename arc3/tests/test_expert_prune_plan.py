"""The prune planner under tools/checkpoints: uniform allocation, the rule's
switches and tables, validation against the checkpoint, the top-k floor, and
the plan file the prune tool consumes.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "checkpoints" / "expert_prune_plan.py"


def _load() -> Any:
    """The tool as a module: tools are scripts, not a package."""
    spec = importlib.util.spec_from_file_location("expert_prune_plan", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: dataclasses resolve the module's deferred
    # annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load()


def _hist(mass: dict[int, list[float]]) -> dict[str, Any]:
    """A histogram with unit counts wherever mass is nonzero."""
    return {
        "captures": 1,
        "counts_by_layer": {str(k): {str(i): 1 for i, v in enumerate(vs) if v} for k, vs in mass.items()},
        "mass_by_layer": {str(k): {str(i): v for i, v in enumerate(vs) if v} for k, vs in mass.items()},
    }


def test_every_layer_gives_up_the_same_count_ranked_within_itself() -> None:
    h = _hist({0: [1, 1, 1, 1, 1, 1, 1, 1], 1: [10, 10, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]})
    plan = P.plan_prune(h, ratio=0.5, rule=P.PruneRule(), layer_ids=[0, 1], num_experts=8, top_k=2)
    assert len(plan.kept[0]) == 4 and len(plan.kept[1]) == 4
    assert plan.kept[1][:2] == [0, 1]  # the two hot experts survive; the rest is the cold tail
    assert plan.kept[1] == sorted(plan.kept[1])
    assert plan.removed_share[1] == pytest.approx(0.04 / 20.06)


def test_floor_keeps_top_k_survivors() -> None:
    h = _hist({0: [1, 1, 1, 1, 1, 1, 1, 1]})
    plan = P.plan_prune(h, ratio=0.9, rule=P.PruneRule(), layer_ids=[0], num_experts=8, top_k=3)
    assert len(plan.kept[0]) == 3
    assert plan.floored == [0]


def test_counts_only_histogram_needs_explicit_counts() -> None:
    h = {"captures": 1, "counts_by_layer": {"0": {"0": 5, "1": 1}}}
    with pytest.raises(ValueError, match="counts only"):
        P.plan_prune(h, ratio=0.5, rule=P.PruneRule(), layer_ids=[0], num_experts=2, top_k=1)
    plan = P.plan_prune(h, ratio=0.5, rule=P.PruneRule(by="counts"), layer_ids=[0], num_experts=2, top_k=1)
    assert plan.kept[0] == [0]


def test_mismatched_model_is_refused() -> None:
    h = _hist({0: [1, 1], 1: [1, 1]})
    with pytest.raises(ValueError, match="do not match"):
        P.plan_prune(h, ratio=0.5, rule=P.PruneRule(), layer_ids=[0], num_experts=2, top_k=1)
    h2 = _hist({0: [1, 1, 1]})
    with pytest.raises(ValueError, match="outside"):
        P.plan_prune(h2, ratio=0.5, rule=P.PruneRule(), layer_ids=[0], num_experts=2, top_k=1)


def test_zero_ratio_keeps_everything() -> None:
    h = _hist({0: [1, 1, 1, 1]})
    plan = P.plan_prune(h, ratio=0.0, rule=P.PruneRule(), layer_ids=[0], num_experts=4, top_k=1)
    assert plan.kept[0] == [0, 1, 2, 3]


def _norm_hist(counts: list[int], gate_norm: list[float], gate_norm_sq: list[float]) -> dict[str, Any]:
    """A one-layer histogram with the norm tables a norm-mode run writes."""
    return {
        "captures": 1,
        "counts_by_layer": {"0": {str(i): c for i, c in enumerate(counts) if c}},
        "gate_norm_by_layer": {"0": {str(i): v for i, v in enumerate(gate_norm) if v}},
        "gate_norm_sq_by_layer": {"0": {str(i): v for i, v in enumerate(gate_norm_sq) if v}},
    }


def test_gate_norm_total_and_conditional_mean_rank_differently() -> None:
    # expert 0: called 100 times at a small contribution each; expert 1: called
    # once with a large one. In total expert 0 delivered more; per activation
    # expert 1 did. The two criteria must disagree on which to keep.
    h = _norm_hist(counts=[100, 1, 1, 1], gate_norm=[10.0, 5.0, 0.1, 0.1], gate_norm_sq=[1.0, 25.0, 0.01, 0.01])
    kw: dict[str, Any] = dict(ratio=0.75, layer_ids=[0], num_experts=4, top_k=1)
    assert P.plan_prune(h, rule=P.PruneRule(norm=True), **kw).kept[0] == [0]
    assert P.plan_prune(h, rule=P.PruneRule(norm=True, conditional=True), **kw).kept[0] == [1]
    assert P.plan_prune(h, rule=P.PruneRule(norm=True, square=True, conditional=True), **kw).kept[0] == [1]


def test_norm_criteria_need_a_norm_histogram() -> None:
    h = _hist({0: [1, 1]})
    with pytest.raises(ValueError, match="ULLM_EXPERT_NORMS"):
        P.plan_prune(h, ratio=0.5, rule=P.PruneRule(norm=True), layer_ids=[0], num_experts=2, top_k=1)
    with pytest.raises(ValueError, match="must be 'mass' or 'counts'"):
        P.PruneRule(by="norms")
    with pytest.raises(ValueError, match="takes no"):
        P.PruneRule(by="counts", conditional=True)


def test_the_cli_writes_the_plan_the_prune_tool_reads(tmp_path: pathlib.Path) -> None:
    """Kept lists ascending, one count for every layer, the source count beside them, provenance."""
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"text_config": {"num_experts": 4, "num_experts_per_tok": 1}}))
    # Two decoder routers and the MTP block's, which is not a decoder layer.
    routers = [f"model.language_model.layers.{i}.mlp.gate.weight" for i in (0, 1)] + ["mtp.layers.0.mlp.gate.weight"]
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in routers}})
    )
    hist = tmp_path / "routed_experts.1.json"
    hist.write_text(json.dumps({**_hist({0: [4, 3, 2, 1], 1: [1, 2, 3, 4]}), "num_experts": 4}))
    out = tmp_path / "plan.json"
    subprocess.run(
        [sys.executable, str(TOOL), "--histogram", str(hist), "--src", str(src), "--ratio", "0.5", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    plan = json.loads(out.read_text())
    assert plan["kept"] == {"0": [0, 1], "1": [2, 3]}
    assert plan["num_experts"] == 4 and plan["num_kept"] == 2 and plan["rule"] == "mass"
    assert plan["histogram_sha256"] and plan["floored"] == []
    # a histogram from another checkpoint is refused by its recorded count
    hist.write_text(json.dumps({**_hist({0: [1, 1], 1: [1, 1]}), "num_experts": 2}))
    failed = subprocess.run(
        [sys.executable, str(TOOL), "--histogram", str(hist), "--src", str(src), "--ratio", "0.5", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "measured on 2 experts" in failed.stderr
    # a histogram covering fewer layers than the checkpoint has routers is
    # refused too: the layer set comes from the checkpoint, not the histogram
    hist.write_text(json.dumps({**_hist({0: [4, 3, 2, 1]}), "num_experts": 4}))
    failed = subprocess.run(
        [sys.executable, str(TOOL), "--histogram", str(hist), "--src", str(src), "--ratio", "0.5", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0 and "do not match the model's MoE layers [0, 1]" in failed.stderr
