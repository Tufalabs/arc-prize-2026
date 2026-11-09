"""Same-checkpoint parity across actual updates, without loading a large model."""

import copy
import random
from dataclasses import replace

import pytest
import torch

from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import build_samples
from training.episode_packing import pack_episode_samples
from test_ppo import _test_state_critic
from test_ppo import _make_tiny_ppo_trainer


def _trainer(monkeypatch):
    trainer = _make_tiny_ppo_trainer(advantage=1.0, learning_rate=0.1)
    trainer._dry_optimizer_step_count = 0
    trainer.config.value_baseline = "hidden_critic"
    trainer.state_critic = _test_state_critic()
    trainer._extract_policy_hidden_states = lambda _rollouts: {}
    original_signals = trainer._compute_step_signals

    def fixed_test_signals(rollouts):
        signals, stats, batch = original_signals(rollouts)
        # This test isolates policy-version parity across real optimizer
        # updates. A one-episode centered window is otherwise degenerate;
        # baseline normalization itself is tested with multiple episodes.
        return (
            [
                [
                    replace(signal, advantage=float(rollout.trajectory_reward))
                    for signal in per_step
                ]
                for rollout, per_step in zip(rollouts, signals, strict=True)
            ],
            stats,
            batch,
        )

    monkeypatch.setattr(trainer, "_compute_step_signals", fixed_test_signals)
    trainer.config.episode_packing = True
    trainer.config.loss_aggregation = "token_mean"
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda root, iteration: (root / "adapter", root / "runtime", 0.0),
    )
    monkeypatch.setattr(
        trainer, "_save_training_state", lambda root, **_kw: root / "state"
    )
    return trainer


def _capture(trainer, policy, reward=1.0):
    """Freeze sampled token scores now; later updates must not alter them."""
    values = torch.log_softmax(trainer.model.logits.detach(), dim=-1).tolist()
    return RolloutTrajectory(
        trajectory_id=f"policy-{policy}",
        task_id="game",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                prompt_token_ids=[1],
                completion_token_ids=[2],
                reward=0.0,
                done=False,
                completion_logprobs=[values[2]],
                metadata={"segment_start": True},
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                prompt_token_ids=[1, 2, 4],
                completion_token_ids=[3, 2],
                completion_logprobs=[values[3], values[2]],
                reward=reward,
                done=True,
            ),
        ],
        trajectory_reward=reward,
        metadata={"policy_iteration": policy, "start_level": 0},
    )


def _samples(trainer, rollout):
    signals, _, _ = trainer._compute_step_signals([rollout])
    samples = build_samples([rollout], per_rollout_signals=signals)
    return pack_episode_samples(
        samples, max_sequence_tokens=8, loss_aggregation="token_mean"
    )[0]


def _assert_tree_equal(before, after):
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _assert_tree_equal(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for a, b in zip(before, after, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert before == after


def _two_updates(trainer, root):
    first = _capture(trainer, -1)
    overlap = _capture(trainer, -1, reward=2.0)
    initial = trainer.model.logits.detach().clone()
    initial_result = trainer.train_window([first], output_dir=root, iteration_index=0)
    learned_capture = _capture(trainer, 0, reward=3.0)
    learned = trainer.model.logits.detach().clone()
    assert not torch.equal(initial, learned)
    overlap_result = trainer.train_window([overlap], output_dir=root, iteration_index=1)
    assert not torch.equal(learned, trainer.model.logits)
    return initial_result, overlap_result, learned_capture, learned


def test_base_and_trained_parity_follow_real_overlapping_updates(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    first, overlap, captured, learned = _two_updates(trainer, tmp_path)
    assert first.rollout_logprob_parity_token_count == 3
    assert first.rollout_logprob_parity_collection_lag == 0
    assert overlap.rollout_logprob_parity_sampler_policy_iteration == -1
    assert overlap.rollout_logprob_parity_trainer_policy_iteration == 0
    assert overlap.rollout_logprob_parity_collection_lag == 1

    # Comparing these captured learned-policy scores against current weights
    # really would fail. The parity check must use the retained checkpoint.
    samples = _samples(trainer, captured)
    wrong = trainer._check_rollout_logprob_parity(samples, trainer_policy_iteration=0)
    assert trainer._rollout_logprob_parity_failure(wrong, sample_count=len(samples))
    before = trainer.model.logits.detach().clone()
    result = trainer.train_window([captured], output_dir=tmp_path, iteration_index=2)
    assert result.rollout_logprob_parity_sampler_policy_iteration == 0
    assert result.rollout_logprob_parity_trainer_policy_iteration == 1
    assert result.rollout_logprob_parity_collection_lag == 1
    assert result.rollout_logprob_parity_checked_policy_lag == 0
    assert result.rollout_logprob_parity_mean_abs == 0.0
    assert result.optimizer_steps == 1
    assert not torch.equal(before, trainer.model.logits)
    assert list(trainer._parity_policy_snapshots) == [0, 1]
    assert torch.equal(trainer._parity_policy_snapshots[0]["logits"], learned)


def test_parity_total_time_includes_snapshot_and_marks_startup(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr("training.ppo.time.perf_counter", lambda: clock[0])
    remember = trainer._remember_parity_policy
    forward = trainer._token_logprobs

    def timed_snapshot(policy):
        remember(policy)
        clock[0] += 7.0

    def timed_forward(batch):
        result = forward(batch)
        clock[0] += 3.0
        return result

    monkeypatch.setattr(trainer, "_remember_parity_policy", timed_snapshot)
    monkeypatch.setattr(trainer, "_token_logprobs", timed_forward)
    result = trainer.train_window(
        [_capture(trainer, -1)], output_dir=tmp_path, iteration_index=0
    )
    assert result.rollout_logprob_parity_startup is True
    assert result.rollout_logprob_parity_snapshot_seconds == 7.0
    assert result.rollout_logprob_parity_forward_seconds == 6.0
    assert result.rollout_logprob_parity_seconds == 13.0
    assert result.rollout_logprob_parity_batch_count == 2


@pytest.mark.parametrize("failure", [None, "captured_logprobs", "forward"])
def test_temporary_scoring_restores_all_training_state(monkeypatch, tmp_path, failure):
    trainer = _trainer(monkeypatch)
    _, _, captured, _ = _two_updates(trainer, tmp_path)
    trainer._remember_parity_policy(1)
    samples = _samples(trainer, captured)
    if failure == "captured_logprobs":
        for sample in samples:
            sample.old_token_logprobs = [
                value + (0.5 if mask else 0.0)
                for value, mask in zip(
                    sample.old_token_logprobs, sample.completion_mask[1:], strict=True
                )
            ]
    original_forward = trainer._token_logprobs

    def forward(batch):
        torch.rand(
            3
        )  # Even a stochastic scoring implementation cannot consume training RNG.
        if failure == "forward":
            raise RuntimeError("injected forward failure")
        return original_forward(batch)

    monkeypatch.setattr(trainer, "_token_logprobs", forward)

    class NoCollectives:
        def __getattr__(self, name):
            pytest.fail(f"parity attempted DDP operation {name}")

    trainer._ddp_group = NoCollectives()
    trainer.model.eval()
    trainer.model.logits.grad = torch.arange(5, dtype=torch.float32)
    parameters = list(trainer.model.parameters())
    values = [parameter.detach().clone() for parameter in parameters]
    gradients = [parameter.grad for parameter in parameters]
    gradient_values = [
        gradient.clone() if gradient is not None else None for gradient in gradients
    ]
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    counters = (trainer._optimizer_step_count, trainer._dry_optimizer_step_count)
    modes = [module.training for module in trainer.model.modules()]
    requires_grad = [parameter.requires_grad for parameter in parameters]
    rng, python_rng = torch.get_rng_state(), random.getstate()

    if failure:
        with pytest.raises(
            RuntimeError, match="parity check failed|injected forward failure"
        ):
            trainer._check_window_logprob_parity(
                samples, trainer_policy_iteration=1, policy_lag_windows=1
            )
    else:
        metrics = trainer._check_window_logprob_parity(
            samples, trainer_policy_iteration=1, policy_lag_windows=1
        )
        assert metrics["token_count"] == 3 and metrics["mean_abs"] == 0.0
    assert all(
        a is b for a, b in zip(parameters, trainer.model.parameters(), strict=True)
    )
    for parameter, value, gradient, gradient_value in zip(
        parameters, values, gradients, gradient_values, strict=True
    ):
        assert torch.equal(parameter, value)
        assert parameter.grad is gradient
        _assert_tree_equal(gradient_value, parameter.grad)
    _assert_tree_equal(optimizer, trainer.optimizer.state_dict())
    assert counters == (
        trainer._optimizer_step_count,
        trainer._dry_optimizer_step_count,
    )
    assert modes == [module.training for module in trainer.model.modules()]
    assert requires_grad == [parameter.requires_grad for parameter in parameters]
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()


def test_zero_advantage_window_still_checks_captured_tokens(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    result = trainer.train_window(
        [_capture(trainer, -1, reward=0.0)], output_dir=tmp_path, iteration_index=0
    )
    assert result.optimizer_steps == 0 and result.sample_count == 0
    assert result.filtered_zero_signal_sample_count == 1
    assert result.rollout_logprob_parity_token_count == 3
    assert result.rollout_logprob_parity_checked_policy_lag == 0


def test_bad_learned_capture_stops_before_baseline_update_optimizer_or_export(
    monkeypatch, tmp_path
):
    trainer = _trainer(monkeypatch)
    _, _, captured, _ = _two_updates(trainer, tmp_path)
    captured.steps[0].completion_logprobs[0] += 0.5
    before = trainer.model.logits.detach().clone()
    optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    baseline = copy.deepcopy(trainer.state_critic.state_dict())
    steps = trainer._optimizer_step_count
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda *args: pytest.fail("a failing parity check exported a policy"),
    )
    with pytest.raises(
        RuntimeError, match="Same-checkpoint logprob parity check failed"
    ):
        trainer.train_window([captured], output_dir=tmp_path, iteration_index=2)
    assert torch.equal(before, trainer.model.logits)
    _assert_tree_equal(optimizer, trainer.optimizer.state_dict())
    _assert_tree_equal(baseline, trainer.state_critic.state_dict())
    assert trainer._optimizer_step_count == steps


@pytest.mark.parametrize("problem", ["names", "shape", "dtype", "nonfinite"])
def test_invalid_cached_tensors_fail_before_mutating_current_weights(
    monkeypatch, problem
):
    trainer = _trainer(monkeypatch)
    samples = _samples(trainer, _capture(trainer, -1))
    trainer._remember_parity_policy(-1)
    with torch.no_grad():
        trainer.model.logits.add_(0.25)
    trainer._remember_parity_policy(0)
    old = trainer._parity_policy_snapshots[-1]
    if problem == "names":
        old["other"] = old.pop("logits")
    elif problem == "shape":
        old["logits"] = old["logits"][:1]
    elif problem == "dtype":
        old["logits"] = old["logits"].double()
    else:
        old["logits"][0] = float("nan")
    before = trainer.model.logits.detach().clone()
    with pytest.raises(RuntimeError, match="parity snapshot"):
        trainer._check_window_logprob_parity(
            samples, trainer_policy_iteration=0, policy_lag_windows=1
        )
    assert torch.equal(before, trainer.model.logits)


@pytest.mark.parametrize(
    "problem, message",
    [
        ("mixed", "mixed rollout policy"),
        ("future", "newer than the trainer"),
        ("lag", "disagrees with the raw samples"),
        ("stale", "at most one"),
        ("missing", "no cached weights"),
    ],
)
def test_incoherent_policy_versions_fail_closed(monkeypatch, problem, message):
    trainer = _trainer(monkeypatch)
    samples = _samples(trainer, _capture(trainer, -1))
    trainer._remember_parity_policy(0)
    reported_lag = 1
    if problem == "mixed":
        samples.append(replace(samples[0], policy_iteration=0))
    elif problem == "future":
        samples[0].policy_iteration = 1
    elif problem == "lag":
        reported_lag = 0
    elif problem == "stale":
        samples[0].policy_iteration = -2
    with pytest.raises(RuntimeError, match=message):
        trainer._check_window_logprob_parity(
            samples, trainer_policy_iteration=0, policy_lag_windows=reported_lag
        )


def test_resumed_first_window_needs_only_its_current_snapshot(monkeypatch, tmp_path):
    trainer = _trainer(monkeypatch)
    with torch.no_grad():
        trainer.model.logits.copy_(torch.tensor([0.0, 0.2, 0.4, 0.1, -0.1]))
    captured = _capture(trainer, 20)
    result = trainer.train_window([captured], output_dir=tmp_path, iteration_index=21)
    assert list(trainer._parity_policy_snapshots) == [20]
    assert result.rollout_logprob_parity_sampler_policy_iteration == 20
    assert result.rollout_logprob_parity_collection_lag == 0
