"""The default baseline remains separate from PPO scheduling and loss weights."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.config import ExperimentConfig, TrainingConfig
from training.ppo import PPOTrainer, _TRAINER_STATE_VERSION

from test_group_baseline import _rollout


ROOT = Path(__file__).resolve().parents[1]


def _payload():
    return json.loads((ROOT / "configs/dppo_2gpu_arc3_ar25.json").read_text())


def _load(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return ExperimentConfig.load(path)


@pytest.mark.parametrize("baseline", ["group_mean"])
def test_critic_disabled_requires_two_attempts_per_game(tmp_path, baseline):
    payload = _payload()
    payload["training"]["value_baseline"] = baseline
    payload["parallel_rollouts"] = len(payload["inference"]["games"])
    with pytest.raises(ValueError, match="at least two rollouts per game"):
        _load(tmp_path, payload)


def test_group_mean_refuses_advantage_clipping_but_opt_in_critic_keeps_it():
    payload = {"model_name": "m", "output_dir": "runs", "advantage_clip": 1}
    with pytest.raises(ValueError, match="requires advantage_clip=0"):
        TrainingConfig.from_dict(payload)
    assert (
        TrainingConfig.from_dict(
            {**payload, "value_baseline": "hidden_critic"}
        ).advantage_clip
        == 1
    )
    assert (
        TrainingConfig.from_dict(
            {"model_name": "m", "output_dir": "runs", "value_baseline": "hidden_critic"}
        ).advantage_clip
        == 5
    )


def test_group_route_never_extracts_hidden_states_or_updates_historical_estimates():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(value_baseline="group_mean")
    trainer.level_baseline = trainer.state_critic = None
    trainer._extract_policy_hidden_states = lambda _: pytest.fail(
        "critic forward called"
    )
    rollouts = [_rollout(0, steps=2), _rollout(100, steps=3)]
    signals, stats, batch = trainer._compute_step_signals(rollouts)
    assert [signal.advantage for signal in signals[0]] == [-0.5, -0.5]
    assert [signal.advantage for signal in signals[1]] == [0.5, 0.5, 0.5]
    assert stats.mean_advantage == 0
    update, seconds = trainer._update_baseline(batch)
    assert update.update_steps == seconds == 0
    assert update.mean_absolute_error == 50
    assert trainer._baseline_feature_seconds == 0
    assert trainer.level_baseline is trainer.state_critic is None


@pytest.mark.parametrize(
    "saved,current",
    [
        ("hidden_critic", "group_mean"),
        ("level_ema", "group_mean"),
        ("group_mean", "hidden_critic"),
    ],
)
def test_baseline_switch_rejects_checkpoint_before_mutating_optimizer(
    tmp_path, saved, current
):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(value_baseline=current)
    trainer.optimizer = SimpleNamespace(
        load_state_dict=lambda _: pytest.fail("optimizer mutated")
    )
    path = tmp_path / "state.pt"
    torch.save(
        {
            "version": _TRAINER_STATE_VERSION,
            "training_config": {"value_baseline": saved},
            "optimizer_state_dict": {},
        },
        path,
    )
    with pytest.raises(RuntimeError, match="group_mean baseline change"):
        trainer.load_training_state(path)


def test_same_group_baseline_checkpoint_restores_without_critic_state(tmp_path):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(value_baseline="group_mean")
    trainer.state_critic = trainer.level_baseline = None
    loaded = []
    trainer.optimizer = SimpleNamespace(
        load_state_dict=lambda state: loaded.append(state)
    )
    path = tmp_path / "state.pt"
    torch.save(
        {
            "version": _TRAINER_STATE_VERSION,
            "training_config": {"value_baseline": "group_mean"},
            "optimizer_state_dict": {"marker": 1},
            "optimizer_step_count": 7,
        },
        path,
    )
    trainer.load_training_state(path)
    assert loaded == [{"marker": 1}]
    assert trainer._optimizer_step_count == 7
