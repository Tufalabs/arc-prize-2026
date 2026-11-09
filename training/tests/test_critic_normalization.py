"""PopArt invariants: raw actor units, sparse-game fallback, and exact resume."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from training.config import TrainingConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.critic_normalization import PerGameTargetNormalizer
from training.state_critic import (
    StateCriticBaseline,
    StateCriticObservation,
    compute_state_critic_step_signals,
)


def _observations(game="a", targets=(10.0, 30.0)):
    return [
        StateCriticObservation(
            rollout_index=index,
            step_index=0,
            game_id=game,
            reward=target,
            score_so_far=0,
            return_to_go=target,
            levels_remaining=1,
            total_levels=2,
            analysis_step=1,
            max_steps=100,
            policy_hidden=[float(index), 1, 2, 3],
        )
        for index, target in enumerate(targets)
    ]


def _critic(**kwargs):
    return StateCriticBaseline(
        **{
            "game_vocab": ["a", "b", "unseen"],
            "game_embedding_buckets": 16,
            "hidden_dim": 8,
            "game_embedding_dim": 2,
            "policy_hidden_dim": 4,
            "policy_hidden_projection_dim": 4,
            "target_normalization": "per_game",
            "normalization_min_rollouts": 2,
            "normalization_alpha": 0.5,
            "initial_value": 0.0,
            **kwargs,
        }
    )


def test_window_moments_use_rollout_counts_and_game_balanced_pool():
    normalizer = PerGameTargetNormalizer(
        ["a", "b", "unseen"],
        initial_scale=100,
        alpha=0.5,
        min_rollouts=2,
        min_std=1,
    )
    # A hundred decisions in one rollout cannot warm up a game's statistics.
    normalizer.update([("a", 0, 10.0)] * 100 + [("b", 1, 90.0)])
    assert normalizer.games["a"].rollout_count == 1
    assert normalizer.parameters("a") == pytest.approx((50, 40, 1))
    assert normalizer.parameters("unseen") == pytest.approx((50, 40, 1))
    normalizer.update([("a", 0, 30.0)])  # New window: rollout indices restart.
    assert normalizer.games["a"].rollout_count == 2
    assert normalizer.parameters("a") == pytest.approx((20, 10, 2))
    assert normalizer.parameters("unknown") == normalizer.parameters("unseen")
    normalizer.update([("a", 0, 20.0), ("a", 1, 20.0)])
    assert normalizer.games["a"].variance == pytest.approx(50)


def test_zero_history_and_constant_targets_have_safe_scales():
    normalizer = PerGameTargetNormalizer(
        ["a"],
        initial_scale=100,
        alpha=1,
        min_rollouts=2,
        min_std=1,
    )
    normalizer.update([("a", index, 0.0) for index in range(100)])
    assert normalizer.parameters("a") == (0, 100, 0)
    normalizer.update([("a", 0, 10.0), ("a", 1, 10.0)])
    assert normalizer.parameters("a") == (10, 1, 2)
    assert normalizer.parameters("unknown") == (10, 1, 1)


def test_stats_changes_preserve_all_raw_values_and_head_optimizer_moments():
    critic = _critic(initial_value=20)
    observations = _observations("a") + _observations("b", (40, 80))
    critic.update(observations, epochs=2, batch_size=2, max_grad_norm=10)
    probes = observations + _observations("unseen") + _observations("unknown")
    # Probe additional levels, horizons and hidden states, not just fitted rows.
    probes += [
        replace(row, level_index=2, policy_hidden=[4, 3, 2, 1]) for row in probes
    ]
    before_values = critic.predict(probes)
    head = critic.network.base[-1]
    before_weights = head.weight.detach().clone()
    before_bias = head.bias.detach().clone()
    before_adam = copy.deepcopy(critic.optimizer.state[head.weight])
    games = [""] + list(critic.game_vocab)
    old = [critic.target_parameters(game) for game in games]

    critic._update_target_normalization(_observations("a", (1, 3)))
    new = [critic.target_parameters(game) for game in games]
    ratio = torch.tensor([a[1] / b[1] for a, b in zip(old, new, strict=True)])
    shift = torch.tensor([(a[0] - b[0]) / b[1] for a, b in zip(old, new, strict=True)])
    assert critic.predict(probes) == pytest.approx(before_values, abs=2e-5)
    torch.testing.assert_close(head.weight, before_weights * ratio[:, None])
    torch.testing.assert_close(head.bias, before_bias * ratio + shift)
    torch.testing.assert_close(
        critic.optimizer.state[head.weight]["exp_avg"],
        before_adam["exp_avg"] * ratio[:, None],
    )
    torch.testing.assert_close(
        critic.optimizer.state[head.weight]["exp_avg_sq"],
        before_adam["exp_avg_sq"] * ratio[:, None].square(),
    )


def test_fallback_to_game_transition_preserves_values_and_raw_actor_advantages():
    critic = _critic(initial_value=7)
    critic._update_target_normalization(_observations("a", (0, 100)))
    rollouts = [
        RolloutTrajectory(
            trajectory_id=f"episode-{index}",
            task_id=game,
            steps=[RolloutStep(0, [], [], {}, reward, True, [1, 2], [3, 4])],
            trajectory_reward=reward,
        )
        for index, (game, reward) in enumerate((("a", 100), ("b", 10), ("b", 30)))
    ]

    def signals(normalize):
        return compute_state_critic_step_signals(
            rollouts,
            context_window_tokens=100,
            critic=critic,
            normalize_advantages=normalize,
            policy_hidden_by_step={(i, 0): [0, 1, 2, 3] for i in range(3)},
        )

    before = signals(False)
    normalized_before = signals(True)
    critic._update_target_normalization(_observations("b", (10, 30)))
    assert critic.target_parameters("b") == (20, 10, 2)
    assert critic.target_parameters("a") == (50, 50, 2)
    for old, new in ((before, signals(False)), (normalized_before, signals(True))):
        assert [row[0].advantage for row in new.per_rollout_signals] == pytest.approx(
            [row[0].advantage for row in old.per_rollout_signals],
            abs=2e-5,
        )
    raw = signals(False)
    assert raw.per_rollout_signals[1][0].advantage == pytest.approx(
        10 - critic.predict(raw.observations)[1],
        abs=2e-5,
    )
    assert before.diagnostics["per_game/b/normalization/source"] == 1
    assert raw.diagnostics["per_game/b/normalization/source"] == 2


def test_mse_uses_historical_normalization_and_updates_moments_once_after_fit():
    critic = _critic(learning_rate=0)
    observations = _observations()
    # First fit: targets / 100 and a zero prior. Stats become ready only after fit.
    update = critic.update(observations, epochs=4, batch_size=2, max_grad_norm=10)
    assert update.loss == pytest.approx(0.05)
    assert critic.target_parameters("a") == (20, 10, 2)
    assert critic.target_normalizer.games["a"].rollout_count == 2
    assert critic.predict(observations) == pytest.approx([0, 0], abs=1e-5)
    # Raw squared error / 10^2, not / 100^2 or a per-minibatch std.
    update = critic.update(observations, epochs=3, batch_size=1, max_grad_norm=10)
    assert update.loss == pytest.approx(5)
    assert update.mean_absolute_error == pytest.approx(20)
    assert critic.target_normalizer.games["a"].rollout_count == 4


def test_games_with_different_reward_scales_have_comparable_critic_losses():
    critic = _critic(learning_rate=0)
    observations = _observations("a", (0, 2)) + _observations("b", (0, 200))
    critic._update_target_normalization(observations)
    tensors = critic._tensorize(observations)
    losses = (critic._predict_tensor(tensors) - tensors["normalized_targets"]).square()
    torch.testing.assert_close(losses[:2], losses[2:])
    assert losses.mean().item() == pytest.approx(2)
    assert critic.predict(observations) == pytest.approx([0] * 4)


def test_pre_fit_metrics_do_not_reuse_post_fit_predictions():
    critic = _critic(policy_hidden_dim=0, learning_rate=0.1)
    rollouts = [
        RolloutTrajectory(
            trajectory_id=str(i),
            task_id="a",
            steps=[RolloutStep(0, [], [], {}, reward, True, [1], [2])],
            trajectory_reward=reward,
        )
        for i, reward in enumerate((10, 30))
    ]
    batch = compute_state_critic_step_signals(
        rollouts, critic=critic, context_window_tokens=100
    )
    metrics = copy.deepcopy(batch.diagnostics)
    prefix = "per_game/a/pre_update/"
    assert metrics[prefix + "bias"] == -20
    assert metrics[prefix + "rmse"] == pytest.approx(500**0.5)
    assert metrics[prefix + "residual_std"] == 10
    assert metrics[prefix + "rollouts"] == 2
    critic.update(batch.observations, epochs=4, batch_size=2, max_grad_norm=10)
    assert critic.predict(batch.observations) != [0, 0]
    assert batch.diagnostics == metrics


def test_checkpoint_restores_statistics_predictions_and_next_optimizer_update(tmp_path):
    critic = _critic()
    observations = _observations("a") + _observations("b", (40, 80))
    critic.update(observations, epochs=2, batch_size=2, max_grad_norm=10)
    path = tmp_path / "critic.pt"
    torch.save(critic.state_dict(), path)
    restored = _critic()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert (
        critic.target_normalizer.state_dict() == restored.target_normalizer.state_dict()
    )
    assert critic.predict(observations) == restored.predict(observations)
    rng = torch.get_rng_state()
    critic.update(observations, epochs=2, batch_size=2, max_grad_norm=10)
    torch.set_rng_state(rng)
    restored.update(observations, epochs=2, batch_size=2, max_grad_norm=10)
    for key, expected in critic.network.state_dict().items():
        assert torch.equal(restored.network.state_dict()[key], expected)
    assert (
        critic.target_normalizer.state_dict() == restored.target_normalizer.state_dict()
    )


def test_fixed_mode_retains_version_7_resume_and_rejects_silent_mode_conversion():
    critic = _critic(target_normalization="fixed")
    state = critic.state_dict()
    state["version"] = 7
    del state["target_normalization"], state["target_normalizer"]
    restored = _critic(target_normalization="fixed", seed=1)
    restored.load_state_dict(state)
    assert critic.predict(_observations()) == restored.predict(_observations())
    with pytest.raises(RuntimeError, match="target_normalization mismatch"):
        _critic().load_state_dict(state)
    with pytest.raises(RuntimeError, match="target_normalization mismatch"):
        restored.load_state_dict(_critic().state_dict())


@pytest.mark.parametrize(
    "field,value", [("alpha", 0.1), ("min_std", 2), ("min_rollouts", 4)]
)
def test_checkpoint_rejects_changed_normalization_settings(field, value):
    state = _critic().state_dict()
    state["target_normalizer"][field] = value
    with pytest.raises(ValueError, match="checkpoint.*mismatch"):
        _critic().load_state_dict(state)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_checkpoint_moments_are_fatal(bad):
    state = _critic().state_dict()
    state["target_normalizer"]["games"]["a"]["variance"] = bad
    with pytest.raises(ValueError, match="normalization moments"):
        _critic().load_state_dict(state)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_targets_do_not_partially_update_running_statistics(bad):
    critic = _critic()
    before = copy.deepcopy(critic.state_dict())
    with pytest.raises(RuntimeError, match="nonfinite"):
        critic._update_target_normalization(_observations(targets=(10, bad)))
    assert critic.target_normalizer.state_dict() == before["target_normalizer"]
    for key, expected in critic.network.state_dict().items():
        assert torch.equal(before["network_state_dict"][key], expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_normalization", "unknown"),
        ("normalization_alpha", 0),
        ("normalization_alpha", float("nan")),
        ("normalization_alpha", 1.1),
        ("normalization_min_std", 0),
        ("normalization_min_std", float("inf")),
        ("normalization_min_rollouts", 1),
        ("normalization_min_rollouts", 2.5),
        ("normalization_min_rollouts", True),
    ],
)
def test_normalization_config_rejects_invalid_settings(field, value):
    with pytest.raises(ValueError, match="state_critic_"):
        TrainingConfig.from_dict(
            {"model_name": "m", "output_dir": "runs", f"state_critic_{field}": value}
        )


def test_normalization_is_optional_and_requires_hidden_critic():
    payload = {"model_name": "m", "output_dir": "runs"}
    assert (
        TrainingConfig.from_dict(payload).state_critic_target_normalization == "fixed"
    )
    payload["state_critic_target_normalization"] = "per_game"
    payload["value_baseline"] = "hidden_critic"
    assert (
        TrainingConfig.from_dict(payload).state_critic_target_normalization
        == "per_game"
    )
    with pytest.raises(ValueError, match="requires value_baseline"):
        TrainingConfig.from_dict({**payload, "value_baseline": "group_mean"})
    with pytest.raises(ValueError, match="requires a game vocabulary"):
        _critic(game_vocab=[])


def test_trainer_builds_per_game_critic_from_config(monkeypatch):
    from test_ppo import _make_tiny_ppo_trainer

    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer.game_vocab = ("a", "b")
    trainer.config.state_critic_target_normalization = "per_game"
    trainer.config.state_critic_normalization_alpha = 0.25
    trainer.config.state_critic_normalization_min_rollouts = 8
    trainer.config.state_critic_normalization_min_std = 2
    monkeypatch.setattr(trainer, "_policy_hidden_size", lambda: 4)
    critic = trainer._build_state_critic()
    assert critic.network.base[-1].out_features == 3
    assert critic.target_normalizer.alpha == 0.25
    assert critic.target_normalizer.min_rollouts == 8
    assert critic.target_normalizer.min_std == 2
