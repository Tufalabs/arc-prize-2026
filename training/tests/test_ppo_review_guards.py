"""Regression checks for optimizer admission and observational independence."""

from collections import defaultdict
from contextlib import nullcontext
import math
from types import SimpleNamespace

import pytest
import torch

from test_ppo import _make_tiny_ppo_trainer, _tiny_action_sample


def test_extra_clip_guard_includes_worker_metrics_before_mutating_weights():
    trainer = _make_tiny_ppo_trainer(advantage=1)
    events = []
    trainer._ddp_group = SimpleNamespace(
        send_cycle=lambda *a, **k: events.append("send"),
        reduce_grads_into_params=lambda: events.append("reduce"),
        gather_worker_metrics=lambda: (
            events.append("metrics") or [{"clip_fraction_sum": 1.0, "micro_batches": 1}]
        ),
        broadcast_weights=lambda: events.append("broadcast"),
    )
    before = [p.detach().clone() for p in trainer._trainable_parameters()]
    accumulators = defaultdict(float)
    fraction = trainer._execute_train_cycle(
        [[_tiny_action_sample(advantage=1)], [_tiny_action_sample(advantage=1)]],
        cycle_index=1,
        epoch_label="extra",
        epoch_number=2,
        iteration_index=0,
        loss_denominator=2,
        world_size=2,
        mem=SimpleNamespace(region=lambda *a, **k: nullcontext()),
        accumulators=accumulators,
        expected_optimizer_steps_total=1,
        max_clip_fraction=0.1,
    )
    assert fraction == 0.5
    assert events == ["send", "reduce", "metrics", "broadcast"]
    assert trainer.optimizer_step_count == accumulators["global_step"] == 0
    assert not trainer.optimizer.state
    for actual, expected in zip(trainer._trainable_parameters(), before, strict=True):
        assert torch.equal(actual, expected)
        assert actual.grad is None


@pytest.mark.parametrize(
    "failure", [torch.OutOfMemoryError("OOM"), RuntimeError("CUDA device-side assert")]
)
@pytest.mark.parametrize("probe", ["entropy", "grad_ratio"])
def test_device_failure_in_observability_remains_fatal(monkeypatch, failure, probe):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    sample = _tiny_action_sample(advantage=1)
    trainer._probe_samples = [sample]
    trainer.live_entropy_coef = 0.1

    def fail(*a, **k):
        raise failure

    monkeypatch.setattr(trainer, "_token_logprobs_and_entropies", fail)
    with pytest.raises(type(failure), match=str(failure)):
        if probe == "entropy":
            trainer._compute_probe_entropy()
        else:
            trainer._entropy_to_policy_grad_ratio([sample])


def test_bad_probe_shape_is_unavailable_instead_of_false_zero_entropy(monkeypatch):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer._probe_samples = [_tiny_action_sample(advantage=1)]

    def fail(*a, **k):
        raise ValueError("invalid shape")

    monkeypatch.setattr(trainer, "_token_logprobs_and_entropies", fail)
    assert math.isnan(trainer._compute_probe_entropy())


def test_trust_region_measures_window_update_not_existing_sampler_lag():
    trainer = _make_tiny_ppo_trainer(advantage=1)
    sample = _tiny_action_sample(advantage=1)
    # Deliberately off-policy rollout evidence must not reject an unchanged
    # current policy: the guard measures change from the window-start policy.
    sample.old_token_logprobs = [-100.0]
    before = trainer._trust_region_token_logprobs([sample])
    assert trainer._post_update_clip_fraction([sample], before) == 0
    with torch.no_grad():
        trainer.model.logits[2] += 2
    assert trainer._post_update_clip_fraction([sample], before) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_trust_probe_disables_dropout_and_restores_modes_and_rng(monkeypatch, fail):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer.config.disable_dropout = False
    trainer.model.probe_dropout = torch.nn.Dropout(p=0.75)
    trainer.model.train()
    trainer.model.model.eval()  # Preserve a deliberately mixed module tree.
    modes = [(module, module.training) for module in trainer.model.modules()]
    rng = torch.get_rng_state().clone()

    def score(batch):
        assert not any(module.training for module in trainer.model.modules())
        torch.rand(5)  # Even an incidental random draw must not affect training.
        if fail:
            raise RuntimeError("probe failed after consuming randomness")
        values = torch.ones_like(batch["input_ids"][:, 1:], dtype=torch.float32)
        functional = torch.nn.functional.dropout(
            values, p=0.5, training=trainer.model.training
        )
        return trainer.model.probe_dropout(functional)

    monkeypatch.setattr(trainer, "_token_logprobs", score)
    sample = _tiny_action_sample(advantage=1)
    if fail:
        with pytest.raises(RuntimeError, match="after consuming randomness"):
            trainer._trust_region_token_logprobs([sample])
    else:
        before = trainer._trust_region_token_logprobs([sample])
        assert torch.equal(before, torch.ones(1))
        assert trainer._post_update_clip_fraction([sample], before) == 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert [(module, module.training) for module in trainer.model.modules()] == modes
    assert trainer.model.probe_dropout.p == 0.75


def test_extra_passes_reshuffle_without_pinning_largest_batch(monkeypatch):
    from training import ppo

    trainer = _make_tiny_ppo_trainer(advantage=1)
    largest, shortest = (
        _tiny_action_sample(advantage=1),
        _tiny_action_sample(advantage=-1),
    )
    monkeypatch.setattr(
        trainer, "_batch_iter", lambda samples: iter([[largest], [shortest]])
    )
    orders = iter([True, False])

    def shuffle(items):
        if next(orders):
            items.reverse()

    monkeypatch.setattr(ppo.random, "shuffle", shuffle)
    first = trainer._extra_pass_cycles([largest, shortest], cycle_size=1)
    second = trainer._extra_pass_cycles([largest, shortest], cycle_size=1)
    assert first[0][0][0] is shortest
    assert second[0][0][0] is largest


def test_rollout_token_metrics_ignore_training_subsample(monkeypatch, tmp_path):
    from training.contracts import RolloutStep, RolloutTrajectory

    trainer = _make_tiny_ppo_trainer(advantage=1, learning_rate=1e-5)
    samples = [_tiny_action_sample(advantage=1) for _ in range(4)]
    rollouts = []
    for i, sample in enumerate(samples):
        sample.trajectory_id = str(i)
        rollouts.append(
            RolloutTrajectory(
                str(i),
                "g",
                [
                    RolloutStep(
                        0,
                        [],
                        [],
                        {},
                        1,
                        True,
                        prompt_token_ids=sample.input_ids[:1],
                        completion_token_ids=sample.input_ids[1:],
                    )
                ],
            )
        )
    monkeypatch.setattr("training.ppo.build_samples", lambda *a, **k: samples)
    monkeypatch.setattr(
        trainer, "_export_policy_snapshot", lambda *a: (tmp_path, tmp_path, 0.0)
    )
    monkeypatch.setattr(
        trainer, "_save_training_state", lambda *a, **k: tmp_path / "state.pt"
    )
    result = trainer.train_window(
        rollouts, output_dir=tmp_path, iteration_index=0, sample_budget_fraction=0.5
    )
    assert result.sample_count == 2
    # Completion tokens per rollout over the whole raw window, not the subsample.
    assert result.avg_tokens_per_rollout == result.avg_tokens_per_model_turn == 1
    assert result.processed_tokens == 4
