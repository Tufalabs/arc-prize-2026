"""Config schema rules that stay fatal: no legacy shims, no silent drops."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.config import ExperimentConfig


@pytest.mark.parametrize("value", [1, 0, -1, True, False, 1.5, 6.0, "6", [], {}])
def test_checkpoint_keep_last_requires_null_or_an_integer_at_least_two(value):
    from training.config import TrainingConfig

    payload = _shipped_recipe_payload()["training"]
    payload["checkpoint_keep_last"] = value
    with pytest.raises(
        ValueError,
        match=r"training\.checkpoint_keep_last must be null or an integer >= 2; "
        r"the prior trainer state must survive the publish/evaluate window "
        r"before training_state\.json advances",
    ):
        TrainingConfig.from_dict(payload)


@pytest.mark.parametrize("value", [None, 2, 6, 12])
def test_checkpoint_keep_last_defaults_to_six_and_round_trips(tmp_path, value):
    from dataclasses import asdict

    from training.config import TrainingConfig

    payload = _shipped_recipe_payload()
    payload["training"].pop("checkpoint_keep_last", None)
    assert TrainingConfig.from_dict(payload["training"]).checkpoint_keep_last == 6
    payload["training"]["checkpoint_keep_last"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    loaded = asdict(ExperimentConfig.load(path).training)
    assert loaded["checkpoint_keep_last"] == value
    assert TrainingConfig.from_dict(loaded).checkpoint_keep_last == value


def test_old_adapter_retention_field_is_unknown(tmp_path):
    payload = _shipped_recipe_payload()
    payload["training"]["adapter_checkpoint_keep_last"] = None
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError,
        match=r"Unknown configuration field\(s\): training\.adapter_checkpoint_keep_last",
    ):
        ExperimentConfig.load(path)


@pytest.mark.parametrize("value", [-0.1, True, "0.1", None, float("nan"), float("inf")])
def test_malformed_call_penalty_requires_a_nonnegative_finite_number(value):
    from training.config import TrainingConfig

    payload = _shipped_recipe_payload()["training"]
    payload["malformed_call_penalty"] = value
    with pytest.raises(ValueError, match="malformed_call_penalty"):
        TrainingConfig.from_dict(payload)


@pytest.mark.parametrize("value", [0, 0.05, 0.2])
def test_malformed_call_penalty_round_trips_in_recipe(tmp_path, value):
    from dataclasses import asdict

    payload = _shipped_recipe_payload()
    payload["training"]["malformed_call_penalty"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = ExperimentConfig.load(path)
    assert asdict(config.training)["malformed_call_penalty"] == value


def test_there_is_no_separate_empty_reply_coefficient(tmp_path):
    """An empty reply is one malformed call (training.malformed_call_penalty);
    a config naming a coefficient of its own is refused, not ignored."""
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/grpo_3gpu_arc2_overfit.json").read_text()
    )
    payload["training"]["empty_reply_penalty"] = 0.05
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="empty_reply_penalty"):
        ExperimentConfig.load(path)


@pytest.mark.parametrize("value", [0, -0.5, True, "0.5", float("nan"), float("inf")])
def test_max_episode_penalty_requires_null_or_a_positive_finite_number(value):
    from training.config import TrainingConfig

    payload = _shipped_recipe_payload()["training"]
    payload["max_episode_penalty"] = value
    with pytest.raises(ValueError, match="max_episode_penalty"):
        TrainingConfig.from_dict(payload)


@pytest.mark.parametrize("value", [None, 0.5, 1, 2.0])
def test_max_episode_penalty_defaults_to_no_cap_and_round_trips(tmp_path, value):
    from dataclasses import asdict

    from training.config import TrainingConfig

    payload = _shipped_recipe_payload()["training"]
    payload.pop("max_episode_penalty", None)
    assert TrainingConfig.from_dict(payload).max_episode_penalty is None
    recipe = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/grpo_3gpu_arc2_overfit.json").read_text()
    )
    recipe["training"]["max_episode_penalty"] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(recipe))
    loaded = asdict(ExperimentConfig.load(path).training)["max_episode_penalty"]
    assert loaded == value and (value is None or type(loaded) is float)


@pytest.mark.parametrize("value", [0.5, 2.0])
def test_arc3_refuses_the_arc2_penalty_cap(tmp_path, value):
    """ARC-3 rewards carry no reply penalties: its configs leave the ARC-2
    cap unset (null is the same), and any value is refused."""
    payload = _shipped_recipe_payload()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    ExperimentConfig.load(path)
    payload["training"]["max_episode_penalty"] = None
    path.write_text(json.dumps(payload))
    ExperimentConfig.load(path)
    payload["training"]["max_episode_penalty"] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training.max_episode_penalty is an ARC-2 reward cap"):
        ExperimentConfig.load(path)


def _shipped_recipe_payload() -> dict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "ppo_2gpu_arc3_games25_platypus.json"
        ).read_text()
    )


def test_missing_completion_cap_is_fatal(tmp_path: Path) -> None:
    """The reply cap sets compaction spacing; no config may omit it."""
    payload = _shipped_recipe_payload()
    del payload["inference"]["agent_max_completion_tokens"]
    config_path = tmp_path / "capless.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="agent_max_completion_tokens is required"):
        ExperimentConfig.load(config_path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("inference", "game"),
        ("inference", "model"),
        ("inference", "allow_runtime_lora_packed_module_warnings"),
        ("training", "load_in_4bit"),
        ("training", "gradient_checkpointing_mode"),
        ("training", "advantage_estimator"),
        ("training", "grpo_std_floor"),
        ("training", "game_baseline_window"),
    ],
)
def test_removed_fields_fail_loudly(tmp_path: Path, section: str, field: str) -> None:
    """A field from an earlier schema is a hard error, never a warning: a
    config states the current schema or the launch stops."""
    payload = _shipped_recipe_payload()
    payload[section][field] = "anything"
    config_path = tmp_path / "stale.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ExperimentConfig.load(config_path)


def test_type_errors_on_live_fields_stay_fatal(tmp_path: Path) -> None:
    payload = _shipped_recipe_payload()
    payload["training"]["learning_rate"] = "not-a-number"
    config_path = tmp_path / "typed.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ExperimentConfig.load(config_path)


def test_shipped_recipe_states_the_stability_defaults() -> None:
    config = ExperimentConfig.load(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ppo_2gpu_arc3_games25_platypus.json"
    )
    assert config.inference.agent_malformed_retries == 0
    assert config.inference.agent_accept_malformed_output is True
    assert config.training.loss_aggregation == "token_mean"
    assert config.training.format_break_penalty == pytest.approx(0.0)


@pytest.mark.parametrize(
    "recipe",
    [
        path
        for path in sorted(
            (Path(__file__).resolve().parents[1] / "configs").glob("*.json")
        )
        if path.name != "sft.json"
    ],
    ids=lambda path: path.stem,
)
def test_shipped_recipes_use_arc_score_without_shaping(recipe):
    config = ExperimentConfig.load(recipe)
    assert config.training.format_break_penalty == 0
    assert config.training.tool_error_penalty == 0
