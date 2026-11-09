"""Dr. GRPO: GRPO without std normalization plus a fixed loss denominator."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from training.config import ExperimentConfig, TrainingConfig
from training.episode_packing import assign_loss_weights
from training.grpo import compute_grpo_signal_batch, compute_grpo_step_signals
from training.ppo import (
    _fixed_token_sum_scale,
    _validate_policy_resume_recipe,
    _window_loss_denominator,
)
from test_grpo import _window


ROOT = Path(__file__).resolve().parents[1]


def _advantages(rewards: list[float], *, normalize_std: bool) -> list[float]:
    signals, _stats = compute_grpo_step_signals(
        _window(rewards),
        group_size=8,
        puzzles_per_window=4,
        normalize_std=normalize_std,
    )
    return [episode[0].advantage for episode in signals[:8]]


def test_dr_grpo_advantage_is_the_group_mean_deviation_without_whitening() -> None:
    rewards = [0.0] * 7 + [-0.05]
    mean = math.fsum(rewards) / 8
    assert _advantages(rewards, normalize_std=False) == pytest.approx(
        [reward - mean for reward in rewards]
    )
    # Scaling every reward in the group scales the advantage; GRPO's
    # std-normalized advantages are invariant to that scaling.
    doubled = [0.0] * 7 + [-0.1]
    assert _advantages(doubled, normalize_std=False) == pytest.approx(
        [value * 2 for value in _advantages(rewards, normalize_std=False)]
    )
    assert _advantages(doubled, normalize_std=True) == pytest.approx(
        _advantages(rewards, normalize_std=True)
    )


def test_grpo_divides_the_same_window_by_the_group_standard_deviation() -> None:
    rewards = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, -0.1]
    mean = math.fsum(rewards) / 8
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / 8)
    centered = _advantages(rewards, normalize_std=False)
    assert centered == pytest.approx([reward - mean for reward in rewards])
    assert _advantages(rewards, normalize_std=True) == pytest.approx(
        [value / (std + 1e-8) for value in centered]
    )


def test_uniform_reward_group_has_zero_advantage_under_both_estimators() -> None:
    assert _advantages([0.25] * 8, normalize_std=False) == [0.0] * 8
    assert _advantages([0.25] * 8, normalize_std=True) == [0.0] * 8


def test_dr_grpo_keeps_the_grpo_step_signal_shape() -> None:
    rewards = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, -0.1]
    rollouts = _window(rewards)
    signals, stats = compute_grpo_step_signals(
        rollouts, group_size=8, puzzles_per_window=4, normalize_std=False
    )
    for rollout, episode in zip(rollouts, signals, strict=True):
        reward = rollout.trajectory_reward
        assert [signal.return_to_go for signal in episode] == [reward] * len(episode)
        assert [signal.reward for signal in episode] == [0.0] * (len(episode) - 1) + [
            reward
        ]
        assert len({signal.advantage for signal in episode}) == 1
    assert stats.rollout_count == 32
    assert stats.mean_reward == pytest.approx(math.fsum(rewards) / 8)


def test_dr_grpo_signal_batch_raw_and_training_advantages_coincide() -> None:
    batch = compute_grpo_signal_batch(
        _window(), group_size=8, puzzles_per_window=4, normalize_std=False
    )
    advantages = [episode[0].advantage for episode in batch.per_rollout_signals]
    assert batch.mean_raw_advantage == pytest.approx(math.fsum(advantages) / 32)
    assert batch.diagnostics["group_count"] == 4.0
    assert batch.diagnostics["group_size"] == 8.0


def test_fixed_token_sum_scale_replaces_the_sampled_token_count() -> None:
    samples = [type("S", (), {"sequence_weight": float(w)})() for w in (300, 500, 200)]
    cycles = 2
    window = _window_loss_denominator(samples, micro_batch_count=4, cycle_size=2)
    assert window == pytest.approx(1000 / cycles)
    scale = _fixed_token_sum_scale(
        rollout_count=4, normalization_tokens=80000, window_token_mass=1000.0
    )
    # Every cycle divides by N * L_ref / cycles, whatever was sampled.
    assert window * scale == pytest.approx(4 * 80000 / cycles)
    assert (
        _fixed_token_sum_scale(
            rollout_count=4, normalization_tokens=80000, window_token_mass=0.0
        )
        == 1.0
    )
    with pytest.raises(ValueError):
        _fixed_token_sum_scale(
            rollout_count=0, normalization_tokens=80000, window_token_mass=1.0
        )
    with pytest.raises(ValueError):
        _fixed_token_sum_scale(
            rollout_count=4, normalization_tokens=0, window_token_mass=1.0
        )


def test_token_sum_fixed_uses_token_mean_weights() -> None:
    class Sample:
        def __init__(self, mask):
            self.completion_mask = mask
            self.trajectory_id = "a"
            self.span_count = 1
            self.sequence_weight = 0.0

    fixed = [Sample([0, 1, 1]), Sample([1, 1, 1, 1])]
    mean = [Sample([0, 1, 1]), Sample([1, 1, 1, 1])]
    assign_loss_weights(fixed, loss_aggregation="token_sum_fixed")
    assign_loss_weights(mean, loss_aggregation="token_mean")
    assert [s.sequence_weight for s in fixed] == [s.sequence_weight for s in mean]
    assert [s.sequence_weight for s in fixed] == [2.0, 4.0]


def test_dr_grpo_recipe_is_the_grpo_recipe_plus_four_overrides() -> None:
    configs = ROOT / "configs"
    assert not (configs / "dr_grpo_2gpu_arc3_ar25.json").exists()
    assert not (configs / "dr_grpo_6gpu_arc2_overfit_lagged.json").exists()
    dr_grpo = json.loads((configs / "dr_grpo_6gpu_arc2_overfit.json").read_text())
    grpo = json.loads((configs / "grpo_6gpu_arc2_overfit.json").read_text())
    grpo["inference"]["agent_terminate_on_empty_reply"] = True
    grpo["training"].update(
        advantage_estimator="dr_grpo",
        loss_aggregation="token_sum_fixed",
        loss_normalization_tokens=80000,
    )
    assert dr_grpo == grpo

    training = ExperimentConfig.load(
        str(configs / "dr_grpo_6gpu_arc2_overfit.json")
    ).training
    assert training.advantage_estimator == "dr_grpo"
    assert training.loss_aggregation == "token_sum_fixed"
    assert training.loss_normalization_tokens == 80000
    # Everything else comes from the ordinary GRPO recipe, unforced.
    assert training.rollout_schedule == "overlapped"
    assert training.extra_train_cycles_until_window is True
    assert training.gradient_accumulation_steps == 8
    assert training.sample_budget.mode == "auto"
    assert (training.clip_ratio_low, training.clip_ratio_high) == (0.3, 0.3)
    assert training.max_window_clip_fraction == 0.5
    assert training.advantage_clip == 0.0
    assert training.value_baseline == "group_mean"
    assert training.policy_loss == "ppo"


def test_dr_grpo_config_validation() -> None:
    base = {
        "model_name": "test",
        "output_dir": "test",
        "advantage_estimator": "dr_grpo",
        "loss_aggregation": "token_sum_fixed",
    }
    config = TrainingConfig.from_dict(base)
    assert config.advantage_estimator == "dr_grpo"
    # No hidden dr_grpo defaults: the knob is the shared normalization constant.
    assert config.loss_normalization_tokens == 32768
    assert config.rollout_schedule == "overlapped"

    with pytest.raises(ValueError, match="advantage_estimator"):
        TrainingConfig.from_dict({**base, "advantage_estimator": "dr-grpo"})
    with pytest.raises(ValueError, match="policy_loss='ppo'"):
        TrainingConfig.from_dict(
            {
                "model_name": "test",
                "output_dir": "test",
                "advantage_estimator": "dr_grpo",
                "policy_loss": "dppo_binary_tv",
            }
        )
    with pytest.raises(ValueError, match="loss_aggregation"):
        TrainingConfig.from_dict({**base, "loss_aggregation": "token_sum"})
    with pytest.raises(ValueError, match="dr_grpo requires loss_aggregation='token_sum_fixed'"):
        TrainingConfig.from_dict(
            {"model_name": "test", "output_dir": "test", "advantage_estimator": "dr_grpo"}
        )
    with pytest.raises(ValueError, match="dr_grpo requires loss_aggregation='token_sum_fixed'"):
        TrainingConfig.from_dict({**base, "loss_aggregation": "token_mean"})
    with pytest.raises(ValueError, match="token_sum_fixed"):
        TrainingConfig.from_dict({**base, "value_baseline": "hidden_critic"})


def _authored_recipe(tmp_path, **training_overrides):
    import json
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    payload = json.loads((configs / "grpo_6gpu_arc2_overfit.json").read_text())
    payload["training"].update(training_overrides)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload))
    return path


def test_normalization_knob_is_rejected_without_a_fixed_denominator(tmp_path) -> None:
    import pytest

    from training.config import ExperimentConfig

    path = _authored_recipe(tmp_path, loss_normalization_tokens=80000)
    with pytest.raises(ValueError, match="loss_normalization_tokens requires"):
        ExperimentConfig.load(str(path))


def test_null_normalization_knob_is_accepted_anywhere(tmp_path) -> None:
    from training.config import ExperimentConfig

    path = _authored_recipe(tmp_path, loss_normalization_tokens=None)
    assert ExperimentConfig.load(str(path)).training.loss_aggregation == "token_mean"


def test_full_config_dump_reloads_with_a_null_knob(tmp_path) -> None:
    from dataclasses import asdict

    from training.config import ExperimentConfig, TrainingConfig

    training = ExperimentConfig.load(str(_authored_recipe(tmp_path))).training
    assert training.loss_normalization_tokens is None
    # The eval worker and gradient replicas reload the full dataclass dump.
    assert TrainingConfig.from_dict(asdict(training)) == training


def test_fixed_denominator_defaults_the_knob(tmp_path) -> None:
    from training.config import ExperimentConfig

    path = _authored_recipe(
        tmp_path, advantage_estimator="dr_grpo", loss_aggregation="token_sum_fixed"
    )
    assert ExperimentConfig.load(str(path)).training.loss_normalization_tokens == 32768


@pytest.mark.parametrize(
    "saved_aggregation,saved_tokens,current_aggregation,current_tokens,changed_field",
    [
        ("token_sum_fixed", 80000, "token_mean", None, "loss_aggregation"),
        ("token_mean", None, "token_sum_fixed", 80000, "loss_aggregation"),
        (
            "token_sum_fixed",
            80000,
            "token_sum_fixed",
            32000,
            "loss_normalization_tokens",
        ),
    ],
)
def test_token_sum_fixed_resume_rejects_changed_objective(
    saved_aggregation, saved_tokens, current_aggregation, current_tokens, changed_field
) -> None:
    def config(aggregation, tokens):
        return TrainingConfig.from_dict(
            {
                "model_name": "test",
                "output_dir": "test",
                "advantage_estimator": (
                    "dr_grpo" if aggregation == "token_sum_fixed" else "grpo"
                ),
                "loss_aggregation": aggregation,
                "loss_normalization_tokens": tokens,
            }
        )

    saved = asdict(config(saved_aggregation, saved_tokens))
    _validate_policy_resume_recipe(saved, config(saved_aggregation, saved_tokens))
    current = config(current_aggregation, current_tokens)
    with pytest.raises(RuntimeError, match=changed_field):
        _validate_policy_resume_recipe(saved, current)
