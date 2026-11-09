import json
from pathlib import Path

import pytest

from training.config import ExperimentConfig, TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,baseline,loss",
    [
        ("dppo_2gpu_arc3_ar25", "group_mean", "sequence_sum_fixed"),
        ("dppo_2gpu_arc3_ar25_hidden_critic", "hidden_critic", "token_mean"),
    ],
)
def test_dppo_recipes_are_full_game_and_single_pass(name, baseline, loss):
    config = ExperimentConfig.load(ROOT / "configs" / f"{name}.json")
    assert config.parallel_rollouts == 25
    assert config.inference.games == ["ar25-0c556536"]
    assert config.training.value_baseline == baseline
    assert config.training.loss_aggregation == loss
    assert config.training.rollout_schedule == "synchronous"
    assert config.training.epochs_per_window == 1
    assert config.training.policy_loss == "dppo_binary_tv"
    assert config.inference.agent_malformed_retries == 0
    assert not hasattr(config.inference, "level_sampling")
    assert not hasattr(config.inference, "summary_compaction")
    assert TrainingConfig.from_dict(config.to_dict()["training"]) == config.training


@pytest.mark.parametrize(
    "field",
    [
        "critic_warmup_iterations",
        "train_on_void_attempts",
        "state_critic_policy_hidden",
    ],
)
def test_obsolete_training_controls_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        TrainingConfig.from_dict({"model_name": "test", "output_dir": "runs", field: 1})


def test_synthetic_benchmark_cannot_bypass_policy_parity(monkeypatch):
    from training.cli import run_training

    monkeypatch.setenv("ARC3_BENCH_SYNTHETIC_WINDOWS", "1")
    with pytest.raises(ValueError, match="mandatory policy-parity"):
        run_training("unused.json")


def test_official25_recipe_preserves_learner_and_nine_disjoint_gpu_roles():
    config = ExperimentConfig.load(
        ROOT / "configs/dppo_9gpu_arc3_official25_hidden_critic_seed61.json"
    )
    provenance = json.loads(
        (ROOT / "docs/recipes/official25_9gpu_20260914.json").read_text()
    )
    historical = provenance["historical_config"]
    # Resolve both through the same schema, including implicit DPPO defaults.
    expected_training = TrainingConfig.from_dict(historical["training"])
    expected_training.lora.dropout = provenance["observed_adapter_dropout"]
    assert config.training == expected_training
    assert config.inference.games == historical["inference"]["games"]
    assert len(set(config.inference.games)) == 25
    assert config.parallel_rollouts == 100
    assert config.parallel_rollouts // config.training.optimizer_minibatch_rollouts == 4
    assert config.training.rollout_schedule == "overlapped"
    assert config.inference.agent_context_window == 32768
    assert config.inference.agent_max_completion_tokens == 8192
    assert config.inference.agent_compact_to_tokens == 14336
    assert config.inference.agent_safety_margin_tokens == 0
    assert config.inference.max_runtime_minutes == 15
    trainer = config.training.device_map_gpus + [
        gpu for worker in config.training.worker_device_map_gpus for gpu in worker
    ]
    rollout = list(map(int, config.inference.server_gpus))
    evaluation = list(map(int, config.official_eval.server_gpus))
    assert [len(trainer), len(rollout), len(evaluation)] == [4, 4, 1]
    assert sorted(trainer + rollout + evaluation) == list(range(9))
    assert config.official_eval.games == config.inference.games
    assert config.official_eval.compact_to_tokens == 33925
    assert provenance["initial_checkpoint"] == 30
    assert provenance["initial_optimizer_steps"] == 124
