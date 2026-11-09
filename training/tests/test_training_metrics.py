"""Timing and work counters must observe the actual optimizer calls."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from training.memory_profile import NullMemoryLogger
from training.ppo import _optimizer_pass_timer
from test_ppo import _make_tiny_ppo_trainer, _single_step_rollout, _tiny_action_sample


def test_micro_batch_times_actual_passes_excluding_loss_construction(monkeypatch):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer.config.reference_kl_coef = 0.1
    clock = [0.0]
    monkeypatch.setattr("training.ppo.time.perf_counter", lambda: clock[0])
    forward = trainer._token_logprobs_and_entropies
    loss_terms = trainer._span_loss_terms

    def actor(batch):
        clock[0] += 2.0
        return forward(batch)

    def reference(batch):
        clock[0] += 5.0
        return torch.zeros_like(batch["old_token_logprobs"])

    def loss(batch, logprobs, entropies, **kwargs):
        clock[0] += 100.0  # Loss construction is outside either pass interval.
        terms = loss_terms(batch, logprobs, entropies, **kwargs)

        def backward(grad):
            clock[0] += 3.0
            return grad

        terms["policy_loss"].register_hook(backward)
        return terms

    monkeypatch.setattr(trainer, "_token_logprobs_and_entropies", actor)
    monkeypatch.setattr(trainer, "_reference_token_logprobs", reference)
    monkeypatch.setattr(trainer, "_span_loss_terms", loss)
    metrics = trainer._micro_batch_forward_backward(
        [_tiny_action_sample(advantage=1)], loss_denominator=1
    )
    assert metrics["processed_tokens"] == 2
    assert metrics["forward_time_s"] == 7.0
    assert metrics["backward_time_s"] == 3.0
    assert any(p.grad is not None for p in trainer._trainable_parameters())


def test_cuda_pass_timer_records_events_without_synchronizing(monkeypatch):
    records = []
    stream = object()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, actual_stream):
            assert actual_stream is stream
            records.append(self)

        def synchronize(self):
            pytest.fail("pass timing must not synchronize per micro-batch")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a: pytest.fail("unexpected sync")
    )
    timings = []
    with _optimizer_pass_timer("cuda:1", timings, "forward_time_s"):
        assert len(records) == 1
        assert timings == []
    assert timings == [("forward_time_s", records[0], records[1])]


def test_window_counts_all_epochs_accumulation_and_unpadded_tokens(
    monkeypatch, tmp_path
):
    trainer = _make_tiny_ppo_trainer(advantage=1, learning_rate=1e-5)
    trainer.config.epochs_per_window = 2
    trainer.config.micro_batch_size = 2
    trainer.config.gradient_accumulation_steps = 2
    samples = [_tiny_action_sample(advantage=1) for _ in range(5)]
    for sample in samples[::2]:
        sample.input_ids.insert(0, 1)
        sample.labels.insert(0, -100)
        sample.attention_mask.insert(0, 1)
        sample.completion_mask.insert(0, 0)
        sample.old_token_logprobs.insert(0, 0.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *a, **k: samples)
    monkeypatch.setattr(
        trainer, "_export_policy_snapshot", lambda *a: (tmp_path, tmp_path, 0.0)
    )
    monkeypatch.setattr(
        trainer, "_save_training_state", lambda *a, **k: tmp_path / "state.pt"
    )
    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )
    assert result.processed_tokens == 2 * (3 * 3 + 2 * 2)
    assert result.optimizer_steps == 4
    assert result.forward_time_s > 0
    assert result.backward_time_s > 0


def test_ddp_work_includes_rejected_cycle_and_uses_rank_phase_maxima(monkeypatch):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer._ddp_group = SimpleNamespace(
        send_cycle=lambda *a, **k: None,
        reduce_grads_into_params=lambda: None,
        gather_worker_metrics=lambda: [
            {
                "clip_fraction_sum": 1,
                "micro_batches": 1,
                "processed_tokens": 17,
                "forward_time_s": 7.0,
                "backward_time_s": 2.0,
            }
        ],
        broadcast_weights=lambda: None,
    )
    monkeypatch.setattr(
        trainer,
        "_micro_batch_forward_backward",
        lambda *a, **k: {
            "mean_ratio": 1.0,
            "clip_fraction": 0.0,
            "entropy": 0.0,
            "entropy_loss": 0.0,
            "loss": 0.0,
            "processed_tokens": 3,
            "forward_time_s": 2.0,
            "backward_time_s": 4.0,
        },
    )
    accumulators = defaultdict(float)
    for _ in range(2):
        trainer._execute_train_cycle(
            [[_tiny_action_sample(advantage=1)]] * 4,
            cycle_index=1,
            epoch_label="extra",
            epoch_number=2,
            iteration_index=0,
            loss_denominator=1,
            world_size=2,
            mem=NullMemoryLogger(),
            accumulators=accumulators,
            expected_optimizer_steps_total=1,
            max_clip_fraction=0.1,
        )
    assert accumulators["processed_tokens"] == 2 * (3 * 2 + 17)
    assert accumulators["forward_time_s"] == 2 * max(2 * 2, 7)
    assert accumulators["backward_time_s"] == 2 * max(4 * 2, 2)
    assert accumulators["global_step"] == 0


def test_cycle_time_excludes_diagnostics_while_optimization_preserves_them(
    monkeypatch, tmp_path
):
    trainer = _make_tiny_ppo_trainer(advantage=1, learning_rate=1e-5)
    clock = [0.0]
    monkeypatch.setattr("training.ppo.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        "training.ppo.build_samples", lambda *a, **k: [_tiny_action_sample(advantage=1)]
    )
    execute = trainer._execute_train_cycle
    parity = trainer._check_window_logprob_parity

    def cycle(*args, **kwargs):
        result = execute(*args, **kwargs)
        clock[0] += 3.0
        return result

    def check_parity(*args, **kwargs):
        clock[0] += 100.0
        return parity(*args, **kwargs)

    def gradient_diagnostic(*args):
        clock[0] += 50.0
        return 0.0

    def probe():
        clock[0] += 200.0
        return 0.0

    def export(*args):
        clock[0] += 400.0
        return tmp_path, tmp_path, 400.0

    monkeypatch.setattr(trainer, "_execute_train_cycle", cycle)
    monkeypatch.setattr(trainer, "_check_window_logprob_parity", check_parity)
    monkeypatch.setattr(trainer, "_entropy_to_policy_grad_ratio", gradient_diagnostic)
    monkeypatch.setattr(trainer, "_compute_probe_entropy", probe)
    monkeypatch.setattr(trainer, "_export_policy_snapshot", export)
    monkeypatch.setattr(
        trainer, "_save_training_state", lambda *a, **k: tmp_path / "state.pt"
    )
    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )
    assert result.optimizer_cycle_seconds == 3.0
    assert result.optimization_seconds == 53.0
    assert result.total_train_window_seconds == 753.0
