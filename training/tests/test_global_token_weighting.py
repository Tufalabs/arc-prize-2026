"""Compare production PPO reductions with an explicit all-token objective."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gloo_ranks import run_rank
from training.dataset import collate_samples
from training.episode_packing import assign_loss_weights, pack_episode_samples
from training.ppo import _DdpWorkerGroup, _window_loss_denominator
from test_episode_packing import _step_sample
from test_ppo import _tiny_loss_test_trainer


def _samples(compact=False):
    first = _step_sample("long", 0, [1, 2], [3, 4], advantage=3.0)
    second = _step_sample(
        "long", 1, first.input_ids + [5], [6, 7, 8, 9, 10, 11], advantage=-1.5
    )
    if compact:
        second = replace(
            second,
            input_ids=[20, 21] + second.input_ids[-6:],
            labels=[-100] * 2 + second.input_ids[-6:],
            attention_mask=[1] * 8,
            completion_mask=[0, 0] + [1] * 6,
            prompt_token_count=2,
            old_token_logprobs=[0.0] * 7,
            segment_start=True,
        )
    short = _step_sample("short", 0, [40, 41, 42, 43], [12], advantage=2.0)
    return [first, second, short]


def _trainer():
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "token_mean"
    trainer.config.previous_policy_kl_coef = 0.2
    trainer.config.reference_kl_coef = 0.0
    trainer.live_entropy_coef = 0.0
    return trainer


def _production_loss(samples, parameter, denominator):
    batch = collate_samples(samples, pad_token_id=0)
    old = batch["old_token_logprobs"]
    new = old + parameter * batch["input_ids"][:, 1:].float() / 100
    terms = _trainer()._span_loss_terms(batch, new, torch.zeros_like(new))
    return (terms["policy_loss"] + terms["previous_policy_kl_loss"]) / denominator


def _dense_loss(parameter):
    features = torch.arange(3, 13, dtype=torch.float32)
    # Token5 is an observation, never part of the objective.
    features = features[features != 5] / 100
    advantages = torch.tensor([3.0] * 2 + [-1.5] * 6 + [2.0])
    log_ratio = parameter * features
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
    kl = log_ratio.exp() - 1 - log_ratio
    return (-surrogate + 0.2 * kl).mean()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("microbatch_size", [1, 2, 3])
def test_global_token_loss_and_gradient_match_dense_reference(
    compact, packed, microbatch_size
):
    samples = _samples(compact)
    assign_loss_weights(samples, loss_aggregation="token_mean")
    if packed:
        samples, _ = pack_episode_samples(
            samples, max_sequence_tokens=64, loss_aggregation="token_mean"
        )
    denominator = _window_loss_denominator(
        samples, micro_batch_count=len(samples), cycle_size=len(samples)
    )
    assert denominator == 9
    parameter = torch.tensor(0.15, requires_grad=True)
    loss = sum(
        _production_loss(samples[i : i + microbatch_size], parameter, denominator)
        for i in range(0, len(samples), microbatch_size)
    )
    (gradient,) = torch.autograd.grad(loss, parameter)
    reference = _dense_loss(parameter)
    (reference_gradient,) = torch.autograd.grad(reference, parameter)
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(gradient, reference_gradient)
    assert abs(gradient.item()) > 0.01


def _rank_gradient(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        samples, _ = pack_episode_samples(
            _samples(True), max_sequence_tokens=64, loss_aggregation="token_mean"
        )
        parameter = torch.nn.Parameter(torch.tensor(0.15))
        denominator = _window_loss_denominator(
            samples, micro_batch_count=3, cycle_size=4
        )
        # Rank0 has three tokens; rank1 has six. No rank-local averaging.
        _production_loss(samples[rank::2], parameter, denominator).backward()
        group = _DdpWorkerGroup.__new__(_DdpWorkerGroup)
        group.parameters = [parameter]
        group.primary_device = torch.device("cpu")
        group._nccl_group = dist.group.WORLD
        group.reduce_grads_into_params()
        if rank == 0:
            Path(output).write_text(json.dumps(parameter.grad.item()))
    finally:
        dist.destroy_process_group()


def test_unequal_ddp_shards_use_the_same_global_token_denominator(tmp_path):
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=run_rank,
            args=(
                rank,
                _rank_gradient,
                str(tmp_path / "rdzv"),
                str(tmp_path / "gradient.json"),
            ),
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=35)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    parameter = torch.tensor(0.15, requires_grad=True)
    (reference,) = torch.autograd.grad(_dense_loss(parameter), parameter)
    assert json.loads((tmp_path / "gradient.json").read_text()) == pytest.approx(
        reference.item()
    )


@pytest.mark.parametrize("same_episode", [False, True])
def test_budget_sampling_preserves_global_token_gradient_in_expectation(
    monkeypatch, same_episode
):
    from training.ppo import _subsample_train_samples

    full = _samples(True)[:2]
    if not same_episode:
        full[1].trajectory_id = "other"
    assign_loss_weights(full, loss_aggregation="token_mean")
    mass = sum(sample.sequence_weight for sample in full)
    parameter = torch.tensor(0.15, requires_grad=True)
    (reference,) = torch.autograd.grad(
        _production_loss(full, parameter, mass), parameter
    )
    gradients = []
    for last in (False, True):
        monkeypatch.setattr(
            "training.ppo.random.sample",
            lambda population, k: list(
                population[-k:] if last and k else population[:k]
            ),
        )
        retained, _ = _subsample_train_samples(
            [replace(sample) for sample in full], fraction=0.5, step_multiple=1
        )
        denominator = _window_loss_denominator(
            retained, micro_batch_count=1, cycle_size=1, expected_mass=mass
        )
        (gradient,) = torch.autograd.grad(
            _production_loss(retained, parameter, denominator), parameter
        )
        gradients.append(gradient)
    torch.testing.assert_close(torch.stack(gradients).mean(), reference)


def test_critic_normalization_counts_only_generated_tokens():
    from types import SimpleNamespace
    from training.state_critic import _normalize_and_clip_advantages

    observations = [
        SimpleNamespace(discount_token_count=count, reward=1.0, rollout_index=i)
        for i, count in enumerate([2, 8, 0])
    ]
    advantages, mean, std = _normalize_and_clip_advantages(
        [10, 0, 100],
        observations,
        normalize_advantages=True,
        advantage_clip=0,
        epsilon=1e-8,
    )
    assert (mean, std) == pytest.approx((2, 4))
    assert advantages[:2] == pytest.approx([2, -0.5])
    assert sum(w * a for w, a in zip([2, 8, 0], advantages)) == pytest.approx(
        0, abs=1e-7
    )
    assert sum(w * a * a for w, a in zip([2, 8, 0], advantages)) / 10 == pytest.approx(
        1
    )
