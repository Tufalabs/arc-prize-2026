import math
from types import SimpleNamespace

import pytest
import torch

from training.dataset import TokenizedSample, collate_samples
from training.memory_profile import NullMemoryLogger
from training.policy_update import (
    binary_tv_mask,
    dppo_binary_tv_objective,
    merge_policy_update_statistics,
    policy_update_statistics,
    summarize_policy_update_statistics,
)
from training.ppo import PPOTrainer, _rollout_minibatches


def _trainer():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        policy_loss="dppo_binary_tv",
        dppo_tv_threshold=0.15,
        dppo_importance_ratio_cap=5.0,
        loss_aggregation="sequence_sum_fixed",
        entropy_coef=0.0,
        previous_policy_kl_coef=0.0,
        reference_kl_coef=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
    )
    return trainer


def _sample(identity, *, advantage=1.0, tokens=2, weight=1.0):
    return TokenizedSample(
        trajectory_id=identity,
        step_index=0,
        input_ids=[1, 2] + [3] * tokens,
        labels=[-100, -100] + [3] * tokens,
        attention_mask=[1] * (2 + tokens),
        completion_mask=[0, 0] + [1] * tokens,
        reward=0.0,
        return_to_go=0.0,
        advantage=advantage,
        prompt_token_count=2,
        old_token_logprobs=[math.log(0.1)] * (1 + tokens),
        sequence_weight=weight,
    )


def test_directional_mask_and_detached_cap_preserve_corrective_gradients():
    old = torch.tensor([0.0001, 0.8, 0.01, 0.8, 0.1, 0.1]).log().requires_grad_()
    current = torch.tensor([0.01, 0.5, 0.1, 0.5, 0.4, 0.1]).log().requires_grad_()
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, 0.0], requires_grad=True)
    objective, mask = dppo_binary_tv_objective(
        current,
        old,
        advantages,
        threshold=0.15,
        ratio_cap=5.0,
    )
    assert mask.tolist() == [True, False, True, True, False, True]
    (-objective.sum()).backward()
    # A 100x rare-token increase remains trainable with gradient capped at 5.
    # Corrective updates also survive even when they are outside the TV bound.
    torch.testing.assert_close(
        current.grad, torch.tensor([-5.0, 0.0, 5.0, -0.625, 0.0, 0.0])
    )
    assert old.grad is None
    assert advantages.grad is None
    assert not mask.requires_grad


def test_tv_boundary_is_inclusive_and_zero_advantage_never_blocks():
    old = torch.tensor([0.25, 0.5, 0.1]).log()
    current = torch.tensor([0.5, 0.25, 0.9]).log()
    assert binary_tv_mask(current, old, torch.tensor([1, -1, 0]), threshold=0.25).all()


def test_extreme_importance_ratio_is_finite_and_has_nonzero_gradient():
    current = torch.tensor([-5.0], requires_grad=True)
    objective, _ = dppo_binary_tv_objective(
        current,
        torch.tensor([-1000.0]),
        torch.ones(1),
        threshold=0.15,
        ratio_cap=5,
    )
    (-objective.sum()).backward()
    assert torch.isfinite(objective).all()
    torch.testing.assert_close(current.grad, torch.tensor([-5.0]))


def test_fixed_token_sum_ignores_segment_weights_and_keeps_masked_denominator():
    trainer = _trainer()
    representations = [
        [_sample("a", tokens=4, weight=100), _sample("quiet", advantage=0, tokens=2)],
        [
            _sample("a", tokens=1, weight=0.01),
            _sample("a", tokens=3, weight=0.99),
            _sample("quiet", advantage=0, tokens=2),
        ],
    ]
    gradients = []
    for samples in representations:
        batch = collate_samples(samples, pad_token_id=0)
        current = batch["old_token_logprobs"].clone().requires_grad_()
        # Block one positive token by an absolute mass shift above .15.
        first_target = (batch["span_ids"][0] > 0).nonzero()[0, 0]
        with torch.no_grad():
            current[0, first_target] = math.log(0.4)
        terms = trainer._span_loss_terms(batch, current, torch.zeros_like(current))
        # Two original rollouts; neither a blocked token nor a quiet episode
        # is removed from the denominator.
        (terms["policy_loss"] / (2 * 32)).backward()
        gradients.append(current.grad[batch["span_ids"] > 0])
    expected = torch.tensor([0, -1, -1, -1, 0, 0], dtype=torch.float32) / 64
    for gradient in gradients:
        torch.testing.assert_close(gradient, expected)


def test_original_rollout_minibatches_preserve_segments_empty_episodes_and_tail():
    ids = [f"episode-{i}" for i in range(25)]
    samples = [_sample(key) for key in ids if key != ids[3]]
    samples += [_sample(ids[0], tokens=1), _sample(ids[0], tokens=3)]
    groups = _rollout_minibatches(samples, rollout_ids=ids, minibatch_rollouts=16)
    assert [count for _, count in groups] == [16, 9]
    assert sum(sample.trajectory_id == ids[0] for sample in groups[0][0]) == 3
    assert all(sample.trajectory_id in ids[16:] for sample in groups[1][0])
    assert sum(len(group) for group, _ in groups) == len(samples)
    assert len({id(sample) for group, _ in groups for sample in group}) == len(samples)
    with pytest.raises(ValueError, match="unique"):
        _rollout_minibatches(
            samples, rollout_ids=[ids[0], ids[0]], minibatch_rollouts=16
        )


def _stats(probabilities, advantages):
    current = torch.tensor(probabilities).log()
    return policy_update_statistics(
        current,
        torch.full_like(current, math.log(0.1)),
        torch.tensor(advantages),
        torch.ones_like(current, dtype=torch.bool),
        tv_threshold=0.15,
        ratio_cap=5,
    )


def test_statistics_pool_counts_and_maxima_instead_of_averaging_worker_means():
    first, second = _stats([0.4], [1.0]), _stats([0.1, 0.1, 0.1], [1.0, 0.0, -1.0])
    merged = {}
    for stats in [first, second]:
        merge_policy_update_statistics(merged, stats)
    summary = summarize_policy_update_statistics(merged)
    assert summary["all/tokens"] == 4
    assert summary["all/mean_abs_change"] == pytest.approx(0.075)
    assert summary["all/max_abs_change"] == pytest.approx(0.3)
    assert summary["all/dppo_masked_fraction"] == 0.25
    assert summary["positive/dppo_masked_fraction"] == 0.5
    assert "negative/p_ge_05/mean_abs_change" not in summary


def test_cycle_merges_rank_zero_and_remote_policy_statistics():
    trainer = _trainer()
    trainer.device = torch.device("cpu")
    trainer._optimizer_step_count = 0
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    parameter.grad = torch.tensor(0.0)
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer._trainable_parameters = lambda: [parameter]
    trainer._set_optimizer_learning_rate = lambda _: 0.1
    local = _stats([0.4], [1.0])
    remote = _stats([0.1, 0.1, 0.1], [1.0, 0.0, -1.0])
    seen = []
    trainer._ddp_group = SimpleNamespace(
        send_cycle=lambda shares, **kw: seen.append(kw["loss_denominator"]),
        reduce_grads_into_params=lambda: None,
        broadcast_weights=lambda: None,
        gather_worker_metrics=lambda: [
            {"micro_batches": 1, "policy_update_statistics": remote}
        ],
    )
    trainer._micro_batch_forward_backward = lambda *_a, **kw: dict(
        mean_ratio=1.0,
        clip_fraction=0.0,
        entropy=0.0,
        entropy_loss=0.0,
        loss=0.0,
        policy_update_statistics=local,
    )
    accumulators = dict(
        mean_ratio_sum=0.0,
        clip_fraction_sum=0.0,
        entropy_sum=0.0,
        entropy_loss_sum=0.0,
        ratio_updates=0,
        grad_norm_sum=0.0,
        max_grad_norm=0.0,
        grad_norm_updates=0,
        global_step=0,
    )
    trainer._execute_train_cycle(
        [[_sample("a")], [_sample("b")]],
        cycle_index=1,
        epoch_label="test",
        epoch_number=1,
        iteration_index=0,
        loss_denominator=64,
        world_size=2,
        mem=NullMemoryLogger(),
        accumulators=accumulators,
        expected_optimizer_steps_total=1,
    )
    summary = summarize_policy_update_statistics(
        accumulators["policy_update_statistics"]
    )
    assert seen == [64]
    assert summary["all/tokens"] == 4
    assert summary["all/dppo_masked_fraction"] == 0.25


@pytest.mark.parametrize("policy_loss", ["ppo", "dppo_binary_tv"])
def test_group_mean_resume_rejects_legacy_ppo_before_loading_optimizer(
    tmp_path, policy_loss
):
    trainer = _trainer()
    trainer.config.policy_loss = policy_loss
    trainer.config.value_baseline = "group_mean"
    called = []
    trainer.optimizer = SimpleNamespace(
        load_state_dict=lambda state: called.append(state)
    )
    path = tmp_path / "old-ppo.pt"
    torch.save(
        {
            "version": 1,
            "training_config": {"policy_loss": "ppo"},
            "optimizer_state_dict": {},
        },
        path,
    )
    with pytest.raises(RuntimeError, match="different training recipe"):
        trainer.load_training_state(path)
    assert called == []


def test_matched_ppo_keeps_ratio_clipping_and_kl_with_fixed_denominator():
    trainer = _trainer()
    trainer.config.policy_loss = "ppo"
    trainer.config.clip_ratio_low = trainer.config.clip_ratio_high = 0.3
    trainer.config.previous_policy_kl_coef = 0.02
    batch = collate_samples(
        [
            _sample("positive", advantage=1, tokens=1, weight=900),
            _sample("negative", advantage=-1, tokens=1, weight=0.001),
        ],
        pad_token_id=0,
    )
    current = torch.full_like(
        batch["old_token_logprobs"], math.log(0.2), requires_grad=True
    )
    terms = trainer._span_loss_terms(batch, current, torch.zeros_like(current))
    ((terms["policy_loss"] + terms["previous_policy_kl_loss"]) / 64).backward()
    # At ratio 2 the positive update is PPO-clipped, the negative update is
    # corrective and remains active. The original k3 KL adds .02 to both.
    torch.testing.assert_close(
        current.grad[batch["span_ids"] > 0],
        torch.tensor([0.02, 2.02]) / 64,
    )


@pytest.mark.parametrize("policy_loss", ["ppo", "dppo_binary_tv"])
def test_group_mean_signal_route_is_stateless_for_both_losses(monkeypatch, policy_loss):
    import training.group_baseline as group_baseline
    from training.baseline_batch import BaselineSignalBatch
    from training.rewards import AdvantageBatchStats

    trainer = _trainer()
    trainer.config.policy_loss = policy_loss
    trainer.config.value_baseline = "group_mean"
    trainer.level_baseline = trainer.state_critic = None
    stats = AdvantageBatchStats(
        rollout_count=0,
        mean_reward=0,
        std_reward=0,
        min_reward=0,
        max_reward=0,
        mean_advantage=0,
        std_advantage=0,
    )
    batch = BaselineSignalBatch(
        per_rollout_signals=[],
        stats=stats,
        rollouts=[],
        mean_value=0,
        mean_target=0,
        mean_raw_advantage=0,
        std_raw_advantage=0,
        explained_variance=0,
        diagnostics={"group_count": 0, "group_size_min": 0, "mean_absolute_error": 0},
    )
    monkeypatch.setattr(
        group_baseline, "build_group_mean_signals", lambda rollouts: batch
    )
    signals, result_stats, result_batch = trainer._compute_step_signals([])
    assert signals == [] and result_stats is stats and result_batch is batch
    update, seconds = trainer._update_baseline(batch)
    assert update.update_steps == 0 and seconds == 0
    assert trainer.level_baseline is trainer.state_critic is None
