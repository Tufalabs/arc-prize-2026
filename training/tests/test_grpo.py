"""Complete-group outcome advantages and critic-free trainer regression tests."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.config import ExperimentConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import build_samples, collate_samples
from training.episode_packing import assign_loss_weights, pack_episode_samples
from training.grpo import compute_grpo_step_signals
from training.ppo import PPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def _window(rewards: list[float] | None = None) -> list[RolloutTrajectory]:
    """Make eight independent episodes for each of four puzzles across servers."""
    values = (
        rewards if rewards is not None else [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, -0.1]
    )
    rollouts = []
    for puzzle in range(4):
        for member, reward in enumerate(values):
            slot = puzzle * 8 + member
            rollouts.append(
                RolloutTrajectory(
                    trajectory_id=f"trajectory-{slot}",
                    task_id=f"puzzle-{puzzle}",
                    done=True,
                    trajectory_reward=reward,
                    metadata={
                        "rollout_slot_id": f"slot-{slot}",
                        "rollout_seed": 1000 + slot,
                        "behavior_policy_snapshot": {
                            "iteration": 3,
                            "adapter_sha256": "abc",
                        },
                        "server_index": member % 2,
                    },
                    steps=[
                        RolloutStep(
                            step_index=0,
                            prompt_messages=[],
                            tool_schemas=[],
                            assistant_message={},
                            reward=0.0,
                            done=False,
                            prompt_token_ids=[1, 2],
                            completion_token_ids=[3, 4],
                            completion_logprobs=[-1.0, -1.5],
                        ),
                        RolloutStep(
                            step_index=1,
                            prompt_messages=[],
                            tool_schemas=[],
                            assistant_message={},
                            reward=reward,
                            done=True,
                            prompt_token_ids=[1, 2, 3, 4, 5, 6],
                            completion_token_ids=[7] * (member + 1),
                            completion_logprobs=[-2.0] * (member + 1),
                        ),
                    ],
                )
            )
    return rollouts


def test_population_advantages_are_constant_across_episode_tokens() -> None:
    """Negative penalties and fractional correctness use the agreed population formula."""
    rollouts = _window()
    signals, stats = compute_grpo_step_signals(
        rollouts, group_size=8, puzzles_per_window=4
    )
    rewards = [rollout.trajectory_reward for rollout in rollouts[:8]]
    mean = sum(rewards) / 8
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / 8)
    for rollout, episode in zip(rollouts, signals, strict=True):
        expected = (rollout.trajectory_reward - mean) / (std + 1e-8)
        assert [signal.advantage for signal in episode] == pytest.approx([expected] * 2)
        assert sum(signal.reward for signal in episode) == rollout.trajectory_reward
        assert [signal.return_to_go for signal in episode] == [
            rollout.trajectory_reward
        ] * 2
    assert stats.mean_reward == pytest.approx(mean)
    assert stats.std_reward == pytest.approx(std)
    assert stats.min_reward == -0.1


@pytest.mark.parametrize("reward", [0.0, 1.0, -0.15])
def test_identical_rewards_have_exact_zero_advantages(reward: float) -> None:
    """All-wrong and equally penalized groups are valid no-signal outcomes."""
    signals, stats = compute_grpo_step_signals(
        _window([reward] * 8), group_size=8, puzzles_per_window=4
    )
    assert all(signal.advantage == 0.0 for episode in signals for signal in episode)
    assert stats.std_advantage == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r.pop(), "Incomplete GRPO"),
        (
            lambda r: setattr(r[0], "task_id", "another-puzzle"),
            "complete puzzle groups",
        ),
        (
            lambda r: r[0].metadata.update(behavior_policy_snapshot="different"),
            "snapshots",
        ),
        (
            lambda r: r[0].metadata.pop("behavior_policy_snapshot"),
            "behavior_policy_snapshot",
        ),
        (
            lambda r: r[0].metadata.update(rollout_seed=r[1].metadata["rollout_seed"]),
            "rollout_seed",
        ),
        (
            lambda r: r[0].metadata.update(
                rollout_slot_id=r[1].metadata["rollout_slot_id"]
            ),
            "rollout_slot_id",
        ),
        (
            lambda r: setattr(r[0], "trajectory_id", r[1].trajectory_id),
            "trajectory identities",
        ),
        (lambda r: setattr(r[0], "done", False), "terminal episodes"),
        (lambda r: setattr(r[0], "trajectory_reward", float("nan")), "finite terminal"),
    ],
)
def test_incomplete_or_mixed_groups_are_rejected(mutation, message: str) -> None:
    """Collection errors cannot silently alter group membership or normalization."""
    rollouts = _window()
    mutation(rollouts)
    with pytest.raises(ValueError, match=message):
        compute_grpo_step_signals(rollouts, group_size=8, puzzles_per_window=4)


def test_packing_retains_generated_token_weight_and_episode_advantages() -> None:
    """Packing changes sequence layout without changing token-mean outcome weighting."""
    rollouts = _window()
    signals, _stats = compute_grpo_step_signals(
        rollouts, group_size=8, puzzles_per_window=4
    )
    samples = build_samples(rollouts, per_rollout_signals=signals)
    assign_loss_weights(samples, loss_aggregation="token_mean")
    expected_mass = sum(sample.sequence_weight for sample in samples)
    expected_advantage_mass = sum(
        sample.sequence_weight * sample.advantage for sample in samples
    )
    packed, _stats = pack_episode_samples(
        samples, max_sequence_tokens=64, tokenizer=None, loss_aggregation="token_mean"
    )
    assert len(packed) == len(rollouts)
    assert sum(sample.sequence_weight for sample in packed) == expected_mass
    batch = collate_samples(packed, pad_token_id=0)
    assert float(
        (batch["token_advantages"] * batch["completion_mask"][:, 1:]).sum()
    ) == pytest.approx(expected_advantage_mass, abs=1e-5)


@pytest.mark.parametrize(
    "variant,train_dataset", [("overfit", "arc2-eval"), ("heldout", "arc2-train")]
)
def test_arc2_recipes_are_independent_fresh_grpo_runs(
    variant: str, train_dataset: str
) -> None:
    """Both launch recipes pin the agreed objective, budgets, and sampling."""
    config = ExperimentConfig.load(ROOT / f"configs/grpo_3gpu_arc2_{variant}.json")
    assert config.inference.environment == "arc2"
    assert config.inference.dataset == train_dataset
    assert config.official_eval.dataset == "arc2-eval"
    assert config.inference.games == config.official_eval.games == []
    assert config.parallel_rollouts == 32
    assert config.sampling.game_sampling_games_per_window == 4
    assert config.training.advantage_estimator == "grpo"
    assert (
        config.training.init_adapter_dir
        is config.training.init_trainer_state_path
        is None
    )
    assert config.inference.agent_max_episode_tokens is None
    assert (
        config.inference.agent_context_window,
        config.inference.agent_max_completion_tokens,
        config.inference.agent_compact_to_tokens,
    ) == (32768, 4096, 28672)
    assert (
        config.inference.max_runtime_minutes
        == config.official_eval.cycle_minutes
        == 15.0
    )
    assert config.official_eval.rollouts_per_game == 1
    assert (
        config.training.previous_policy_kl_coef
        == config.training.reference_kl_coef
        == config.training.entropy_coef
        == 0.0
    )
    assert config.training.loss_aggregation == "token_mean"
    assert config.training.learning_rate == 5e-6


def _trainer(monkeypatch) -> PPOTrainer:
    """Build a real trainer with a CPU policy and fatal critic entry points."""
    config = ExperimentConfig.load(
        ROOT / "configs/grpo_3gpu_arc2_overfit.json"
    ).training
    config = replace(
        config,
        device="cpu",
        device_map_gpus=[],
        bf16=False,
        episode_packing=False,
    )
    monkeypatch.setattr(
        "training.ppo.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(
        PPOTrainer, "_load_policy_model", lambda *_a, **_kw: torch.nn.Linear(2, 2)
    )
    monkeypatch.setattr(
        "training.multimodal.load_processing_stack", lambda *_a, **_kw: object()
    )
    for name in (
        "_build_state_critic",
        "_extract_policy_hidden_states",
    ):
        monkeypatch.setattr(
            PPOTrainer,
            name,
            lambda *_a, **_kw: pytest.fail("GRPO entered a critic path"),
        )
    return PPOTrainer(
        config,
        grpo_group_size=8,
        grpo_puzzles_per_window=4,
        checkpoint_identity={
            "environment": "arc2",
            "advantage_estimator": "grpo",
            "dataset_manifest": {"sha256": "dataset"},
            "processing_stack": {"sha256": "processor"},
        },
    )


@pytest.mark.parametrize("policy_lag", [None, 0])
def test_zero_signal_window_checks_parity_and_skips_critic_and_optimizer(
    monkeypatch, tmp_path: Path, policy_lag: int | None
) -> None:
    """All-wrong startup still validates native scores and exports unchanged policy."""
    trainer = _trainer(monkeypatch)
    assert trainer.state_critic is None
    monkeypatch.setattr(
        trainer, "_token_logprobs", lambda batch: batch["old_token_logprobs"] + 0.001
    )
    monkeypatch.setattr(
        trainer.optimizer, "step", lambda *_a, **_kw: pytest.fail("optimizer ran")
    )
    monkeypatch.setattr(
        trainer, "_export_policy_snapshot", lambda *_a: (tmp_path, tmp_path, 0.0)
    )
    result = trainer.train_window(
        _window([0.0] * 8),
        output_dir=tmp_path,
        iteration_index=0,
        policy_lag_windows=policy_lag,
    )
    assert result.optimizer_skipped_reason == "zero_nonzero_advantage_samples"
    assert result.baseline_update_steps == 0
    assert trainer.optimizer_step_count == 0
    assert result.optimizer_steps == 0
    assert result.rollout_logprob_parity_token_count > 0
    assert result.rollout_logprob_parity_mean_abs == pytest.approx(0.001, abs=1e-6)
    assert result.rollout_logprob_parity_p95_abs == pytest.approx(0.001, abs=1e-6)
    assert result.rollout_logprob_parity_max_abs == pytest.approx(0.001, abs=1e-6)
    assert result.rollout_logprob_parity_signed_mean == pytest.approx(0.001, abs=1e-6)
    assert result.rollout_logprob_parity_outlier_count == 0
    assert result.rollout_logprob_parity_outlier_fraction == 0.0
    state = torch.load(result.trainer_state_path, weights_only=False)
    assert "state_critic_state" not in state
    assert state["checkpoint_identity"] == trainer._checkpoint_identity


def test_zero_signal_parity_failure_refuses_export(monkeypatch, tmp_path: Path) -> None:
    """A zero gradient must not conceal a serving/trainer mismatch."""
    trainer = _trainer(monkeypatch)
    monkeypatch.setattr(
        trainer, "_token_logprobs", lambda batch: batch["old_token_logprobs"] + 0.1
    )
    monkeypatch.setattr(
        trainer.optimizer, "step", lambda *_a, **_kw: pytest.fail("optimizer ran")
    )
    monkeypatch.setattr(
        trainer, "_export_policy_snapshot", lambda *_a: pytest.fail("policy exported")
    )
    with pytest.raises(RuntimeError, match="Same-checkpoint logprob parity check failed"):
        trainer.train_window(_window([0.0] * 8), output_dir=tmp_path, iteration_index=0)
    assert trainer.optimizer_step_count == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("policy_lag", [None, 1])
def test_lagged_zero_signal_window_requires_cached_sampler_policy(
    monkeypatch, tmp_path, policy_lag
):
    """Older-policy draws are checked against their own cached weights, never skipped."""
    trainer = _trainer(monkeypatch)
    monkeypatch.setattr(
        trainer.optimizer, "step", lambda *_a, **_kw: pytest.fail("optimizer ran")
    )
    monkeypatch.setattr(
        trainer, "_export_policy_snapshot", lambda *_a: pytest.fail("policy exported")
    )
    with pytest.raises(RuntimeError, match="no cached weights for sampler policy"):
        trainer.train_window(
            _window([0.0] * 8),
            output_dir=tmp_path,
            iteration_index=1,
            policy_lag_windows=policy_lag,
        )
    assert trainer.optimizer_step_count == 0


def test_grpo_checkpoint_identity_and_algorithm_are_required(
    monkeypatch, tmp_path: Path
) -> None:
    """Resumes reject incompatible objectives, datasets, and processing stacks."""
    trainer = _trainer(monkeypatch)
    state = trainer._training_state_payload(
        iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
    )
    path = tmp_path / "state.pt"
    torch.save(state, path)
    trainer.load_training_state(path)
    for field in (
        "environment",
        "dataset_manifest",
        "processing_stack",
        "malformed_call_penalty",
    ):
        altered = copy.deepcopy(state)
        altered["checkpoint_identity"][field] = "different"
        torch.save(altered, path)
        with pytest.raises(RuntimeError, match="mismatch"):
            trainer.load_training_state(path)
    legacy = copy.deepcopy(state)
    legacy["training_config"].pop("advantage_estimator")
    torch.save(legacy, path)
    with pytest.raises(RuntimeError, match="advantage estimator mismatch"):
        trainer.load_training_state(path)


def test_existing_arc3_recipe_defaults_are_unchanged() -> None:
    """Existing text rollouts retain critic-based PPO without new configuration fields."""
    config = ExperimentConfig.load(
        ROOT / "configs/ppo_3gpu_arc3_synthetic49_heldout_official25.json"
    )
    assert config.inference.environment == "arc3"
    assert config.training.advantage_estimator == "state_critic"
    assert config.inference.dataset is None
    assert config.official_eval.dataset is None


def test_grpo_config_rejects_fractional_groups(tmp_path: Path) -> None:
    """An invalid rollout count fails before collection starts."""
    payload = json.loads((ROOT / "configs/grpo_3gpu_arc2_overfit.json").read_text())
    payload["parallel_rollouts"] = 31
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="divisible"):
        ExperimentConfig.load(path)
