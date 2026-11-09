"""A plan names what its statistics were replayed through and from."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_plans.py"
spec = importlib.util.spec_from_file_location("prune_plans", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _rollout(recording: str, run: str, **fields) -> dict:
    return {"run_id": run, "steps": f"{recording}/traces/{run}/steps.jsonl", **fields}


def test_a_plan_records_the_checkpoint_the_mode_and_the_recordings():
    meta = {
        "a": _rollout("/runs/official", "a", checkpoint="/ckpt/unpruned", multimodal=True),
        "b": _rollout("/runs/hard", "b", checkpoint="/ckpt/unpruned", multimodal=True),
        "c": _rollout("/runs/official", "c", checkpoint="/ckpt/unpruned", multimodal=True),
    }
    assert module.provenance(meta) == {
        "source_checkpoint": "/ckpt/unpruned",
        "replayed_images": True,
        "calibration_recordings": ["/runs/hard", "/runs/official"],
    }


def test_statistics_from_before_the_fields_were_recorded_give_none():
    found = module.provenance({"a": _rollout("/runs/official", "a")})
    assert found["source_checkpoint"] is None and found["replayed_images"] is None
    assert found["calibration_recordings"] == ["/runs/official"]


def test_rollouts_replayed_two_ways_are_not_one_calibration():
    meta = {
        "a": _rollout("/runs/official", "a", checkpoint="/ckpt/unpruned", multimodal=True),
        "b": _rollout("/runs/official", "b", checkpoint="/ckpt/unpruned", multimodal=False),
    }
    with pytest.raises(SystemExit, match="disagree on multimodal"):
        module.provenance(meta)
