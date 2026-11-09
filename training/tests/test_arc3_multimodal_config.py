"""ARC-3 images must reach the policy without changing the tool protocol."""

import json
from pathlib import Path

import pytest

from harness_support.arc3 import trainer as support
from training.config import ExperimentConfig, InferenceConfig
from training.multimodal import build_checkpoint_identity

RECIPE = Path(__file__).resolve().parents[1] / "configs/ppo_2gpu_arc3_games25_platypus.json"


@pytest.mark.parametrize("mode", ["", "current_grid", "level_start"])
def test_image_config_pins_processing_and_observation_mode(monkeypatch, tmp_path, mode):
    payload = json.loads(RECIPE.read_text())
    payload["inference"]["agent_multimodal_context"] = mode
    payload["training"]["value_baseline"] = "group_mean"
    payload["parallel_rollouts"] = 2 * len(payload["inference"]["games"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = ExperimentConfig.load(path)
    stack = {"sha256": "processor"}
    calls = []
    monkeypatch.setattr(
        "training.multimodal.resolve_processing_identity",
        lambda *args: calls.append(args) or stack,
    )
    identity = build_checkpoint_identity(config, support=support, manifest=None)
    assert support.media_requirements(config.inference) == {"images": bool(mode)}
    if mode:
        assert identity["environment"] == "arc3"
        assert identity["multimodal_context"] == mode
        assert identity["processing_stack"] == stack
        assert calls == [(config.training.model_name, config.training.model_revision)]
    else:
        assert identity is None and calls == []


@pytest.mark.parametrize("mode", [None, True, "video", "CURRENT_GRID"])
def test_invalid_image_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="agent_multimodal_context"):
        InferenceConfig.from_dict({"base_url": "http://localhost:1", "agent_multimodal_context": mode})


def test_arc3_image_setting_is_not_an_arc2_override():
    with pytest.raises(ValueError, match="ARC-3 setting"):
        InferenceConfig.from_dict({"base_url": "http://localhost:1", "environment": "arc2", "agent_multimodal_context": "current_grid"})


def test_image_observations_require_a_multimodal_training_model(tmp_path):
    payload = json.loads(RECIPE.read_text())
    payload["inference"]["agent_multimodal_context"] = "current_grid"
    payload["training"]["multimodal"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="multimodal=true"):
        ExperimentConfig.load(path)


@pytest.mark.parametrize("mode", ["", "current_grid", "level_start"])
@pytest.mark.parametrize("baseline", ["group_mean", "hidden_critic"])
@pytest.mark.parametrize("policy_loss", ["ppo", "dppo_binary_tv"])
def test_image_observations_require_a_supported_baseline(
    tmp_path, mode, baseline, policy_loss
):
    recipe = (
        RECIPE
        if policy_loss == "ppo"
        else RECIPE.parent / "dppo_2gpu_arc3_multimodal_smoke.json"
    )
    payload = json.loads(recipe.read_text())
    payload["inference"]["agent_multimodal_context"] = mode
    payload["training"]["value_baseline"] = baseline
    payload["parallel_rollouts"] = 2 * len(payload["inference"]["games"])
    if baseline == "hidden_critic":
        payload["training"]["loss_aggregation"] = "token_mean"
        payload["training"].pop("loss_normalization_tokens", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    if mode and baseline == "hidden_critic":
        with pytest.raises(ValueError, match="critic feature extraction.*image inputs"):
            ExperimentConfig.load(path)
    else:
        config = ExperimentConfig.load(path)
        assert config.training.value_baseline == baseline
        assert config.inference.agent_multimodal_context == mode
