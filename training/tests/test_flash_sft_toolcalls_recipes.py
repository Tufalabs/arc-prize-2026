"""The tool-call-aligned experiments are PR 121's frozen recipe with only data, W&B identity and loss changed."""

import json
from pathlib import Path

import pytest

from scripts.flash_logprob_toolcalls_20261030.configs import EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1] / "configs/flash_sft"


def differences(actual, expected):
    return {k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)}


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_experiment_differs_from_the_mixed_recipe_only_in_data_identity_and_loss(name):
    baseline = json.loads((ROOT / "reap307-kimi-mixed-cosine.json").read_text())
    arm = json.loads((ROOT / f"{name}.json").read_text())
    assert differences(arm, baseline) <= {"rows", "validation", "wandb", "recorded_logprobs"}
    assert differences(arm["validation"], baseline["validation"]) == {"data", "data_sha256"}
    assert differences(arm["wandb"], baseline["wandb"]) == {"name", "id"}
    assert arm["wandb"]["name"] == name and arm["wandb"]["id"]
    assert differences(arm["recorded_logprobs"], baseline["recorded_logprobs"]) <= {"weight", "ce_positions"}
    assert {k: arm["recorded_logprobs"][k] for k in ("weight", "ce_positions")} == EXPERIMENTS[name]
    assert arm["recorded_logprobs"]["require_logprobs"] is False
    assert arm["rows"]["data"] != baseline["rows"]["data"] and len(arm["rows"]["data_sha256"]) == 64
    assert arm["optimizer"]["decay"]["decay_until"] == 483 and arm["passes"] == 2
    assert arm["rows_render_files"] == baseline["rows_render_files"]


def test_experiment_identities_are_distinct():
    ids = {json.loads((ROOT / f"{name}.json").read_text())["wandb"]["id"] for name in EXPERIMENTS}
    assert len(ids) == len(EXPERIMENTS)
