from __future__ import annotations

import copy
import datetime
import logging
import math
import os
import random
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from gloo_ranks import spawn_ranks
from training.dataset import TokenizedSample
from training.ddp_worker import round_robin_split
from training.full_weight_ppo import (
    FullWeightPPOTrainer,
    _fsdp_cycle_plan,
    _slot_imbalance,
    padded_rank_shard,
)
from training.memory_profile import NullMemoryLogger
from training.ppo import _micro_batch_padded_tokens


class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden))


class _TinyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(5, 2)
        self.layers = torch.nn.ModuleList([_TinyDecoderLayer()])

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        hidden = self.embed_tokens(input_ids)
        if attention_mask is None:
            hidden = hidden.mean(dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return (
            hidden
            if attention_mask is None
            else SimpleNamespace(last_hidden_state=hidden)
        )


class _TinyPolicy(torch.nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.model = _TinyDecoder()
        self.lm_head = torch.nn.Linear(2, 5, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens


class _TinyVisionPolicy(_TinyPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.model.visual = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(p=0.5)
        )
        self.config = SimpleNamespace(use_cache=True)


def _training_config(**overrides):
    values = {
        "bf16": False,
        "disable_dropout": True,
        "gradient_checkpointing": False,
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "max_grad_norm": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "entropy_coef": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _weighted_sample() -> TokenizedSample:
    return TokenizedSample(
        trajectory_id="sample",
        step_index=0,
        input_ids=[1, 2, 3, 4],
        labels=[-100, -100, 3, 4],
        attention_mask=[1, 1, 1, 1],
        completion_mask=[0, 0, 1, 1],
        reward=0.0,
        return_to_go=0.0,
        advantage=1.0,
        prompt_token_count=2,
        old_token_logprobs=[0.0, 0.0, 0.0],
        sequence_weight=1.0,
    )


def _loss_only_trainer(score: torch.nn.Parameter) -> FullWeightPPOTrainer:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config()
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._optimizer_step_count = 0

    def token_scores(batch, **_kwargs):
        shape = (batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1)
        logprobs = score.expand(shape)
        return logprobs, torch.zeros_like(logprobs)

    trainer._token_logprobs_and_entropies = token_scores
    return trainer


def test_backward_scale_default_is_neutral_and_zero_keeps_zero_grad() -> None:
    score = torch.nn.Parameter(torch.tensor(0.0))
    trainer = _loss_only_trainer(score)
    sample = _weighted_sample()

    trainer._micro_batch_forward_backward([sample], loss_denominator=2.0)
    default_grad = score.grad.detach().clone()
    score.grad = None
    trainer._micro_batch_forward_backward(
        [sample], loss_denominator=2.0, backward_scale=1.0
    )
    explicit_one_grad = score.grad.detach().clone()
    score.grad = None
    trainer._micro_batch_forward_backward(
        [sample], loss_denominator=2.0, backward_scale=0.0
    )

    assert torch.equal(default_grad, explicit_one_grad)
    assert torch.count_nonzero(default_grad).item() == 1
    assert score.grad is not None
    assert torch.count_nonzero(score.grad).item() == 0


def _micro_batch(name: str, *lengths: int) -> list[TokenizedSample]:
    return [
        replace(_weighted_sample(), trajectory_id=name, input_ids=[1] * length)
        for length in lengths
    ]


def test_cycle_plan_pairs_similar_lengths_and_pads_with_the_smallest() -> None:
    # Padded tokens: a 8, b 2*5=10, c 3, d 12, e 6.
    cycle = [
        _micro_batch("a", 8),
        _micro_batch("b", 5, 4),
        _micro_batch("c", 3),
        _micro_batch("d", 12),
        _micro_batch("e", 6),
    ]
    shares, padding = _fsdp_cycle_plan(cycle, 2)
    names = [[micro[0].trajectory_id for micro in share] for share in shares]
    assert names == [["d", "a", "c"], ["b", "e"]]
    assert padding is cycle[2]
    assert _slot_imbalance(shares) == pytest.approx(
        ((12 - 11) + (8 - 7) + (3 - 1.5)) / (12 + 8 + 3)
    )
    assert _slot_imbalance(round_robin_split(cycle, 2)) == pytest.approx(
        ((10 - 9) + (12 - 7.5) + (6 - 3)) / (10 + 12 + 6)
    )


@pytest.mark.parametrize("world_size", range(1, 9))
def test_cycle_plan_keeps_membership_and_never_pads_above_a_slot(
    world_size: int,
) -> None:
    generator = random.Random(world_size)
    for count in range(1, 2 * world_size + 3):
        cycle = [
            _micro_batch(
                str(index), *[generator.randint(1, 40)] * generator.randint(1, 3)
            )
            for index in range(count)
        ]
        shares, padding = _fsdp_cycle_plan(cycle, world_size)
        assert sorted(map(id, (m for share in shares for m in share))) == sorted(
            map(id, cycle)
        )
        assert max(map(len, shares)) - min(map(len, shares)) <= 1
        assert max(map(len, shares)) == -(-count // world_size)
        size = _micro_batch_padded_tokens
        assert size(shares[0][0]) == max(map(size, cycle))
        assert size(padding) == min(map(size, cycle))
        # The padding micro-batch is no larger than any slot's largest, so a
        # padding rank never lengthens its slot.
        slot_maxima = [
            max(size(share[slot]) for share in shares if slot < len(share))
            for slot in range(len(shares[0]))
        ]
        assert size(padding) <= min(slot_maxima)
        assert _fsdp_cycle_plan(cycle, world_size) == (shares, padding)
        assert (
            _slot_imbalance(shares)
            <= _slot_imbalance(round_robin_split(cycle, world_size)) + 1e-12
        )


def test_full_weight_rank_zero_loader_stages_on_cpu_and_freezes_vision() -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(bf16=True)
    trainer.device = torch.device("cuda")
    model = _TinyVisionPolicy()
    seen_kwargs: dict[str, object] = {}

    def load_base_model(kwargs):
        seen_kwargs.update(kwargs)
        return model

    trainer._load_base_model = load_base_model
    trainer._resolved_attn_implementation = lambda: "sdpa"

    loaded = trainer._load_policy_model()

    assert loaded is model
    assert seen_kwargs == {
        "trust_remote_code": True,
        "dtype": torch.float32,
        "device_map": None,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    assert model.config.use_cache is False
    assert all(
        not parameter.requires_grad for parameter in model.model.visual.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())
    assert model.model.visual[1].p == 0.0


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "device,available,major,initial,expected",
    [
        ("cuda", True, 10, None, "0"),
        ("cuda", True, 10, "1", "0"),
        ("cuda", True, 9, "1", "1"),
        ("cuda", True, 8, None, None),
        ("cpu", False, None, "1", "1"),
        ("cpu", True, 10, "1", "1"),
    ],
)
def test_full_weight_selects_blackwell_backend_before_model_construction(
    monkeypatch, caplog, rank, device, available, major, initial, expected
) -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(bf16=True)
    trainer.device = torch.device(device)
    trainer._resolved_attn_implementation = lambda: "sdpa"
    if initial is None:
        monkeypatch.delenv("FLA_TILELANG", raising=False)
    else:
        monkeypatch.setenv("FLA_TILELANG", initial)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    def capability():
        assert device == "cuda" and available
        return major, 0

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    constructors = []

    def construct(kind):
        assert os.environ.get("FLA_TILELANG") == expected
        constructors.append(kind)
        return _TinyVisionPolicy()

    trainer._load_base_model = lambda _kwargs: construct("checkpoint")
    trainer._load_empty_policy_model = lambda **_kwargs: construct("meta")
    with caplog.at_level("INFO", logger="training.full_weight_ppo"):
        model = trainer._load_policy_model()
    assert constructors == ["checkpoint" if rank == 0 else "meta"]
    assert os.environ.get("FLA_TILELANG") == expected
    assert ("selecting FLA Triton backend" in caplog.text) == (
        device == "cuda" and available and major == 10
    )
    assert all(not value.requires_grad for value in model.model.visual.parameters())
    assert all(value.requires_grad for value in model.lm_head.parameters())


def test_full_weight_loader_rejects_non_fp32_trainable_parameters() -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(bf16=True)
    trainer.device = torch.device("cuda")
    model = _TinyPolicy().to(torch.bfloat16)
    trainer._load_base_model = lambda _kwargs: model
    trainer._resolved_attn_implementation = lambda: "sdpa"

    with pytest.raises(RuntimeError, match="FP32 persistent trainable parameters"):
        trainer._load_policy_model()


def test_full_weight_empty_loader_builds_multimodal_model_on_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(model_name="Qwen/base", multimodal=True)
    trainer._model_revision = lambda: "revision-a"
    seen: dict[str, object] = {}

    def load_config(model_name: str, **kwargs):  # noqa: ANN003
        seen["config"] = (model_name, kwargs)
        return SimpleNamespace()

    def build_model(model_config, **kwargs):  # noqa: ANN001, ANN003
        model = _TinyPolicy().to(dtype=torch.bfloat16)
        seen["model_config"] = model_config
        seen["model_kwargs"] = kwargs
        seen["devices_during_init"] = {
            parameter.device.type for parameter in model.parameters()
        }
        seen["dtypes_during_init"] = {
            parameter.dtype for parameter in model.parameters()
        }
        return model

    monkeypatch.setattr(
        "training.full_weight_ppo.AutoConfig.from_pretrained",
        load_config,
    )
    monkeypatch.setattr(
        "training.full_weight_ppo.AutoModelForImageTextToText.from_config",
        build_model,
    )

    model = trainer._load_empty_policy_model(attn_implementation="sdpa")

    assert seen["config"] == (
        "Qwen/base",
        {"revision": "revision-a", "trust_remote_code": True},
    )
    assert seen["model_kwargs"] == {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    assert seen["devices_during_init"] == {"meta"}
    assert seen["dtypes_during_init"] == {torch.bfloat16}
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_full_weight_probe_entropy_failure_is_not_swallowed() -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(entropy_control=SimpleNamespace(enabled=True))
    trainer._probe_samples = [_weighted_sample()]

    def fail(_probe) -> float:  # noqa: ANN001
        raise RuntimeError("rank-local forward failed")

    trainer._compute_probe_entropy_forward = fail

    with pytest.raises(RuntimeError, match="rank-local forward failed"):
        trainer._compute_probe_entropy()


def test_full_weight_window_finalization_creates_no_lora_artifacts(
    tmp_path: Path,
) -> None:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)

    artifacts = trainer._finalize_window_artifacts(tmp_path, 3)

    assert artifacts.publication_path is None
    assert artifacts.checkpoint_path is None
    assert artifacts.adapter_dir is None
    assert artifacts.policy_export_seconds == 0.0
    assert list(tmp_path.iterdir()) == []


def _cycle_accumulators() -> dict[str, float | int]:
    return {
        "mean_ratio_sum": 0.0,
        "clip_fraction_sum": 0.0,
        "entropy_sum": 0.0,
        "entropy_loss_sum": 0.0,
        "previous_policy_kl_sum": 0.0,
        "reference_kl_sum": 0.0,
        "ratio_updates": 0,
        "grad_norm_sum": 0.0,
        "max_grad_norm": 0.0,
        "grad_norm_updates": 0,
        "final_loss": 0.0,
        "global_step": 0,
    }


def _centralized_expected_update(
    state: dict[str, torch.Tensor],
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    learning_rate: float,
    max_grad_norm: float,
) -> tuple[dict[str, torch.Tensor], float]:
    model = _TinyPolicy()
    model.load_state_dict(state)
    losses = []
    for input_ids, target in examples:
        logits = model.lm_head(model.model(input_ids))
        losses.append(torch.nn.functional.mse_loss(logits, target))
    (sum(losses) / len(losses)).backward()
    parameters = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    grad_norm = float(
        torch.sqrt(
            sum(parameter.grad.float().square().sum() for parameter in parameters)
        ).item()
    )
    clip_coefficient = min(1.0, max_grad_norm / (grad_norm + 1e-6))
    with torch.no_grad():
        for parameter in parameters:
            parameter.grad.mul_(clip_coefficient)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        foreach=False,
    )
    optimizer.step()
    return {
        name: value.detach() for name, value in model.state_dict().items()
    }, grad_norm


def _distributed_uneven_cycle_worker(
    rank: int, world_size: int, rendezvous_path: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        torch.manual_seed(1234)
        reference_policy = _TinyPolicy()
        initial_state = {
            name: value.detach().clone()
            for name, value in reference_policy.state_dict().items()
        }
        if rank == 0:
            policy = reference_policy
        else:
            with torch.device("meta"):
                policy = _TinyPolicy()
            assert all(
                parameter.device.type == "meta" for parameter in policy.parameters()
            )
        config = _training_config()
        # Lengths 3, 2, 4: the cycle's first micro-batch is neither its
        # largest nor its smallest, so rank assignment and padding choice
        # both differ from dealing the cycle in order.
        examples = [
            (
                torch.tensor([[0, 1, 2]], dtype=torch.long),
                torch.tensor([[0.2, -0.1, 0.0, 0.3, -0.2]]),
            ),
            (
                torch.tensor([[2, 3]], dtype=torch.long),
                torch.tensor([[-0.4, 0.1, 0.2, -0.2, 0.5]]),
            ),
            (
                torch.tensor([[1, 4, 0, 3]], dtype=torch.long),
                torch.tensor([[0.3, 0.0, -0.3, 0.4, 0.1]]),
            ),
        ]
        expected_state, expected_grad_norm = _centralized_expected_update(
            initial_state,
            examples,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
        )

        trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.model = policy
        trainer._optimizer_step_count = 0
        trainer._place_policy_model()
        sharded, replicated = trainer._partition_trainable_parameters()
        assert sharded
        assert replicated == list(trainer.model.lm_head.parameters())
        optimizer_with_groups = trainer._build_optimizer()
        assert len(optimizer_with_groups.param_groups) == 2
        trainer.optimizer = optimizer_with_groups

        ran = []

        def micro_batch(
            chunk,
            *,
            loss_denominator,
            backward_scale=1.0,
            **_kwargs,
        ):
            input_ids, target = chunk[0].ids, chunk[0].target
            logits = trainer.model.lm_head(trainer.model.model(input_ids))
            loss = torch.nn.functional.mse_loss(logits, target)
            (loss / loss_denominator * backward_scale).backward()
            marker = float(input_ids[0, 0].item()) / 10.0
            ran.append((input_ids.numel(), backward_scale))
            return {
                "mean_ratio": 1.0,
                "clip_fraction": marker,
                "entropy": 0.5,
                "entropy_loss": 0.0,
                "previous_policy_kl": 0.25 + marker,
                "reference_kl": 0.0,
                "loss": float(loss.detach().item()),
                "spans": 1.0,
                "processed_tokens": input_ids.numel(),
                "forward_time_s": 1.0,
                "backward_time_s": 2.0,
            }

        trainer._micro_batch_forward_backward = micro_batch
        initial_snapshot = trainer._snapshot_update_state()
        cycle = [
            [SimpleNamespace(input_ids=ids[0].tolist(), ids=ids, target=target)]
            for ids, target in examples
        ]
        accumulators = _cycle_accumulators()
        clip_fraction = trainer._execute_train_cycle(
            cycle,
            cycle_index=1,
            epoch_label="test",
            epoch_number=1,
            iteration_index=0,
            loss_denominator=float(len(cycle)),
            world_size=world_size,
            mem=NullMemoryLogger(),
            accumulators=accumulators,
            expected_optimizer_steps_total=1,
        )

        from torch.distributed.tensor import DTensor

        actual_state = {}
        for name, value in trainer.model.state_dict().items():
            actual_state[name] = (
                value.full_tensor().detach()
                if isinstance(value, DTensor)
                else value.detach()
            )
        for name, expected in expected_state.items():
            torch.testing.assert_close(
                actual_state[name], expected, rtol=2e-5, atol=2e-6
            )
        assert float(accumulators["grad_norm_sum"]) == pytest.approx(
            expected_grad_norm, rel=2e-5, abs=2e-6
        )
        # Longest first by rank (4 | 3), then the 2-token micro-batch on rank
        # 0 and, at zero weight, again on rank 1 as the cheapest padding.
        assert ran == ([(4, 2.0), (2, 2.0)] if rank == 0 else [(3, 2.0), (2, 0.0)])
        assert accumulators["processed_tokens"] == 4 + 3 + 2 + 2
        assert accumulators["forward_time_s"] == 2.0
        assert accumulators["backward_time_s"] == 4.0
        assert accumulators["ratio_updates"] == 3
        assert accumulators["previous_policy_kl_sum"] == pytest.approx(1.05)
        assert accumulators["reference_kl_sum"] == 0.0
        assert accumulators["global_step"] == 1
        assert clip_fraction == pytest.approx((0.0 + 0.2 + 0.1) / 3.0)
        assert all(
            parameter.dtype == torch.float32
            for parameter in trainer._trainable_parameters()
        )
        for state in trainer.optimizer.state.values():
            assert state["exp_avg"].dtype == torch.float32
            assert state["exp_avg_sq"].dtype == torch.float32

        gathered_heads = [
            torch.empty_like(trainer.model.lm_head.weight) for _ in range(2)
        ]
        dist.all_gather(gathered_heads, trainer.model.lm_head.weight)
        torch.testing.assert_close(gathered_heads[0], gathered_heads[1])
        assert trainer._synchronize_cycle_seconds(float(rank + 1)) == world_size

        # Reject a second Adam update and restore existing FP32 moments and
        # local shards, without full-tensor CPU snapshots or lost DTensor layouts.
        trainer.config.max_window_clip_fraction = 0.05
        trainer._dry_optimizer_step_count = 2
        snapshot = trainer._snapshot_update_state()
        assert all(backup.device.type == "cpu" for _, backup in snapshot["parameters"])
        assert all(
            backup.numel() < parameter.numel()
            for parameter, backup in snapshot["parameters"]
            if isinstance(parameter, DTensor)
        )
        health_calls = 0

        def health_check():
            nonlocal health_calls
            health_calls += 1

        trainer._execute_train_cycle(
            cycle,
            cycle_index=2,
            epoch_label="reject",
            epoch_number=1,
            iteration_index=0,
            loss_denominator=float(len(cycle)),
            world_size=world_size,
            mem=NullMemoryLogger(),
            accumulators=_cycle_accumulators(),
            expected_optimizer_steps_total=2,
            check_health=health_check,
        )
        assert health_calls == 3  # two equalized slots, then the optimizer step
        assert trainer._optimizer_step_count == 2
        trainer._dry_optimizer_step_count = 3
        assert trainer._apply_window_trust_region(
            snapshot,
            post_update_clip_fraction=0.2,
            iteration_index=0,
        )
        assert trainer._optimizer_step_count == 1
        assert trainer._dry_optimizer_step_count == 2
        assert trainer._trust_region_lr_scale == 0.5
        assert trainer._consecutive_window_rejections == 1
        for parameter, backup in snapshot["parameters"]:
            torch.testing.assert_close(trainer._local_tensor(parameter), backup)
        for parameter, state in snapshot["optimizer_state"].items():
            for name, (template, backup) in state.items():
                restored = trainer.optimizer.state[parameter][name]
                assert isinstance(restored, DTensor) == isinstance(template, DTensor)
                torch.testing.assert_close(trainer._local_tensor(restored), backup)
        assert not trainer._apply_window_trust_region(
            snapshot,
            post_update_clip_fraction=0.0,
            iteration_index=1,
        )
        assert trainer._trust_region_lr_scale == 0.625
        assert trainer._consecutive_window_rejections == 0

        # Rejecting the first-ever update must also remove lazily created Adam
        # state; no stale moments may survive into the next window.
        trainer._restore_update_state(initial_snapshot)
        assert not trainer.optimizer.state
        assert trainer._optimizer_step_count == trainer._dry_optimizer_step_count == 0
        for parameter, backup in initial_snapshot["parameters"]:
            torch.testing.assert_close(trainer._local_tensor(parameter), backup)

        health_calls = 0

        def fail_before_step():
            nonlocal health_calls
            health_calls += 1
            if health_calls == 3:
                raise RuntimeError("serving failed")

        with pytest.raises(RuntimeError, match="serving failed"):
            trainer._execute_train_cycle(
                cycle,
                cycle_index=3,
                epoch_label="health failure",
                epoch_number=1,
                iteration_index=0,
                loss_denominator=float(len(cycle)),
                world_size=world_size,
                mem=NullMemoryLogger(),
                accumulators=_cycle_accumulators(),
                expected_optimizer_steps_total=3,
                check_health=fail_before_step,
            )
        assert not trainer.optimizer.state
        assert trainer._optimizer_step_count == 0
        for parameter, backup in initial_snapshot["parameters"]:
            torch.testing.assert_close(trainer._local_tensor(parameter), backup)
        with pytest.raises(RuntimeError, match="decision differs across ranks"):
            trainer._apply_window_trust_region(
                initial_snapshot,
                post_update_clip_fraction=float(rank),
                iteration_index=0,
            )
    finally:
        dist.destroy_process_group()


def _distributed_probe_state_worker(
    rank: int, world_size: int, rendezvous_path: str, output_dir: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        from training.ppo import PPOTrainer

        trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
        trainer.config = _training_config(entropy_control=SimpleNamespace(enabled=True))
        trainer.device = torch.device("cpu")
        original = PPOTrainer._ensure_probe_states

        def record_disk_owner(self, samples, root):
            Path(root, f"probe_disk_rank_{rank}").write_text("called")
            self._probe_samples = list(samples)

        PPOTrainer._ensure_probe_states = record_disk_owner
        try:
            trainer._ensure_probe_states([_weighted_sample()], Path(output_dir))
        finally:
            PPOTrainer._ensure_probe_states = original

        assert trainer._probe_samples[0].trajectory_id == "sample"

        # The entropy forward stays all-rank, but each rank scores a share of
        # the probe micro-batches and every rank reports the pooled mean.
        trainer.config.micro_batch_size = 1
        trainer.model = torch.nn.Linear(1, 1)
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer._probe_samples = [
            _probe_sample(index, completion_tokens=index + 1) for index in range(5)
        ]
        scored = []

        def entropies(batch):
            ids = batch["input_ids"]
            scored.append(int(ids[0, 0]) - 10)
            values = (ids[:, 1:] * 0 + ids[:, :1] - 9).float()
            return torch.zeros_like(values), values

        trainer._token_logprobs_and_entropies = entropies
        trainer._span_loss_terms = _token_span_entropy
        # Micro-batch shuffling draws on the global Python RNG, which must keep
        # advancing identically on every rank for later training cycles.
        random.seed(5)
        mean = trainer._compute_probe_entropy()
        report = [None] * world_size
        dist.all_gather_object(report, (mean, len(scored), random.random()))
        assert len({value for value, _, _ in report}) == 1
        assert mean == sum((index + 1) ** 2 for index in range(5)) / sum(
            index + 1 for index in range(5)
        )
        assert [forwards for _, forwards, _ in report] == [3, 3]
        assert len({draw for _, _, draw in report}) == 1

        dist.barrier()
        assert Path(output_dir, "probe_disk_rank_0").is_file()
        assert not Path(output_dir, "probe_disk_rank_1").exists()
    finally:
        dist.destroy_process_group()


def _probe_protocol_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
    trainer_path: str,
) -> None:
    """The shared probe protocol (DistributedCycleMixin) for ``module:Trainer``."""
    import importlib

    module, name = trainer_path.split(":")
    trainer_class = importlib.import_module(module).__dict__[name]
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    broadcast = dist.broadcast_object_list
    broadcasts: list[int] = []

    def counted(objects, *args, **kwargs):
        broadcasts.append(len(objects))
        return broadcast(objects, *args, **kwargs)

    def trainer() -> Any:
        result = trainer_class.__new__(trainer_class)
        result.device = torch.device("cpu")
        return result

    dist.broadcast_object_list = counted
    try:
        root = Path(output_dir)
        first = trainer()
        # The first window with samples captures on rank zero, once.
        first._ensure_probe_states([_weighted_sample()], root)
        assert [s.trajectory_id for s in first._probe_samples] == ["sample"]
        assert len(broadcasts) == 1
        # Later windows find every rank ready: no broadcast, no file access.
        path = first._probe_states_path(root)
        stamp = path.stat().st_mtime_ns
        later = replace(_weighted_sample(), trajectory_id="later")
        for _ in range(2):
            first._ensure_probe_states([later], root)
        assert len(broadcasts) == 1 and path.stat().st_mtime_ns == stamp
        assert [s.trajectory_id for s in first._probe_samples] == ["sample"]
        # A resumed trainer loads the file on rank zero and shares it once.
        resumed = trainer()
        resumed._ensure_probe_states([later], root)
        assert [s.trajectory_id for s in resumed._probe_samples] == ["sample"]
        assert len(broadcasts) == 2
        # A rank-zero file error raises on every rank instead of a hang.
        blocked = root / "blocked"
        if rank == 0:
            blocked.write_text("a file where the probe directory belongs")
        dist.barrier()
        with pytest.raises(
            RuntimeError, match="Rank-zero probe state setup failed: FileExistsError"
        ):
            trainer()._ensure_probe_states([_weighted_sample()], blocked)
        dist.barrier()
        Path(output_dir, f"probe_protocol_rank_{rank}").write_text("done")
    finally:
        dist.broadcast_object_list = broadcast
        dist.destroy_process_group()


def run_probe_protocol(tmp_path: Path, trainer_path: str) -> None:
    spawn_ranks(
        _probe_protocol_worker,
        2,
        str(tmp_path / "probe_protocol_init"),
        str(tmp_path),
        trainer_path,
    )
    for rank in (0, 1):
        assert Path(tmp_path, f"probe_protocol_rank_{rank}").is_file()


def test_two_rank_probe_states_broadcast_once_and_relay_rank_zero_errors(
    tmp_path: Path,
) -> None:
    run_probe_protocol(tmp_path, "training.full_weight_ppo:FullWeightPPOTrainer")


def _distributed_structure_mismatch_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        policy = _TinyPolicy()
        if rank == 1:
            policy.register_buffer("peer_only", torch.zeros(1))
        trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
        trainer.config = _training_config()
        trainer.device = torch.device("cpu")
        trainer.model = policy

        with pytest.raises(RuntimeError, match="differs from rank zero"):
            trainer._place_policy_model()
        Path(output_dir, f"structure_mismatch_rank_{rank}").write_text("raised")
    finally:
        dist.destroy_process_group()


def _parity_sample(outputs: int) -> TokenizedSample:
    ids = list(range(1, 5 + outputs))
    return TokenizedSample(
        trajectory_id="parity",
        step_index=0,
        input_ids=ids,
        labels=[-100] * 4 + ids[4:],
        attention_mask=[1] * len(ids),
        completion_mask=[0] * 4 + [1] * outputs,
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
        prompt_token_count=4,
        old_token_logprobs=[0.0] * 3 + [-0.4] * outputs,
    )


# What rank one's own forward does to the sampler's logprobs; rank zero
# always reproduces them exactly. Each case: the verdict on every rank, and
# the scored targets of each parity forward (identical on every rank).
_PARITY_FAULTS = {
    "none": ("passed: 128 targets, not expanded", [128]),
    # Within tolerance alone, rank zero must still fail with rank one.
    "wrong": ("mean_abs=1.75", [128, 1000]),
    "nonfinite": ("1 non-finite completion token", [128]),
    # One 2.5-nat token in 128 expands the check on every rank, and 1 in
    # 1,000 then passes everywhere.
    "outlier": ("passed: 1000 targets, expanded", [128, 1000]),
}


def _distributed_parity_worker(
    rank: int, world_size: int, rendezvous_path: str, output_dir: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        for fault, (verdict, targets) in _PARITY_FAULTS.items():
            trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
            trainer.config = SimpleNamespace(
                micro_batch_size=1,
                max_micro_batch_tokens=65536,
                rollout_logprob_check_batches=1,
            )
            trainer.device = torch.device("cpu")
            trainer.tokenizer = SimpleNamespace(pad_token_id=0)
            trainer.model = torch.nn.Linear(1, 1)
            trainer._set_policy_mode_for_training_forward = lambda: None
            forwards: list[int] = []

            def score(batch, fault=fault, forwards=forwards):
                checked = batch["completion_mask"][:, 1:].bool()
                forwards.append(int(checked.sum()))
                scores = batch["old_token_logprobs"].clone()
                first = tuple(checked.nonzero()[0])
                if rank == 1 and fault == "wrong":
                    scores[checked] += 1.75
                elif rank == 1 and fault == "nonfinite":
                    scores[first] = float("nan")
                elif rank == 1 and fault == "outlier":
                    scores[first] += 2.5
                return scores

            trainer._token_logprobs = score
            trainer._remember_parity_policy(-1)
            try:
                metrics = trainer._check_window_logprob_parity(
                    [_parity_sample(1500)],
                    trainer_policy_iteration=-1,
                    policy_lag_windows=0,
                )
            except RuntimeError as error:
                outcome = str(error)
            else:
                outcome = (
                    f"passed: {metrics['token_count']} targets, "
                    f"{'expanded' if metrics['expanded'] else 'not expanded'}"
                )
            report: list[Any] = [None] * world_size
            dist.all_gather_object(report, (outcome, forwards))
            assert report[0] == report[1], (fault, report)
            assert verdict in outcome and forwards == targets, (fault, outcome, forwards)
        Path(output_dir, f"parity_rank_{rank}").write_text("done")
    finally:
        dist.destroy_process_group()


def test_two_rank_parity_pools_rank_local_errors_into_one_verdict(
    tmp_path: Path,
) -> None:
    """Each rank scores with its own forward; a fault on rank one fails both."""
    spawn_ranks(
        _distributed_parity_worker, 2, str(tmp_path / "parity_init"), str(tmp_path)
    )
    assert Path(tmp_path, "parity_rank_0").is_file()
    assert Path(tmp_path, "parity_rank_1").is_file()


def test_two_rank_uneven_cycle_matches_centralized_clipped_update(
    tmp_path: Path,
) -> None:
    rendezvous_path = str(tmp_path / "fsdp_init")
    spawn_ranks(_distributed_uneven_cycle_worker, 2, rendezvous_path)


def test_two_rank_probe_setup_uses_only_rank_zero_disk_io(tmp_path: Path) -> None:
    spawn_ranks(
        _distributed_probe_state_worker, 2, str(tmp_path / "probe_init"), str(tmp_path)
    )


def test_full_weight_structure_mismatch_fails_on_every_rank(tmp_path: Path) -> None:
    spawn_ranks(
        _distributed_structure_mismatch_worker,
        2,
        str(tmp_path / "structure_init"),
        str(tmp_path),
    )

    assert Path(tmp_path, "structure_mismatch_rank_0").is_file()
    assert Path(tmp_path, "structure_mismatch_rank_1").is_file()


def test_tied_embedding_and_head_are_rejected(tmp_path: Path) -> None:
    rendezvous_path = str(tmp_path / "tied_init")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=0,
        world_size=1,
    )
    try:
        trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
        trainer.config = _training_config()
        trainer.device = torch.device("cpu")
        trainer.model = _TinyPolicy(tied=True)
        with pytest.raises(RuntimeError, match="weight is tied"):
            trainer._place_policy_model()
    finally:
        dist.destroy_process_group()


def test_placement_shards_decoder_but_not_frozen_vision_or_head(
    tmp_path: Path,
) -> None:
    rendezvous_path = str(tmp_path / "placement_init")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=0,
        world_size=1,
    )
    try:
        trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
        trainer.config = _training_config()
        trainer.device = torch.device("cpu")
        trainer.model = _TinyVisionPolicy()
        for parameter in trainer.model.model.visual.parameters():
            parameter.requires_grad_(False)

        trainer._place_policy_model()

        from torch.distributed.tensor import DTensor

        assert isinstance(trainer.model.model.embed_tokens.weight, DTensor)
        assert isinstance(trainer.model.model.layers[0].proj.weight, DTensor)
        assert all(
            not isinstance(parameter, DTensor)
            for parameter in trainer.model.model.visual.parameters()
        )
        assert all(
            not isinstance(parameter, DTensor)
            for parameter in trainer.model.lm_head.parameters()
        )
    finally:
        dist.destroy_process_group()


def _assert_exact(actual, expected) -> None:
    """Nested critic state: tensors bitwise equal, other leaves equal by value."""
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            _assert_exact(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_exact(left, right)
    else:
        assert actual == expected


def _distributed_full_window_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
    reject: bool,
    critic: bool = False,
) -> None:
    from training.config import ExperimentConfig
    from training.contracts import RolloutStep, RolloutTrajectory
    from training.dataset import collate_samples

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        config = ExperimentConfig.load(
            Path(__file__).resolve().parents[1] / "configs/grpo_3gpu_arc2_overfit.json"
        ).training
        config = replace(
            config,
            device="cpu",
            device_map_gpus=[],
            bf16=False,
            multimodal=False,
            gradient_checkpointing=False,
            attn_implementation="sdpa",
            episode_packing=False,
            max_sequence_tokens=64,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            epochs_per_window=1,
            # An oversized, unwarmed step moves the post-update probe past the bound.
            learning_rate=1.0 if reject else 0.01,
            warmup_steps=0,
            max_window_clip_fraction=0.1 if reject else 1.0,
            extra_train_cycles_until_window=not critic,
            extra_cycle_max_clip_fraction=1.0,
            advantage_estimator="state_critic" if critic else "grpo",
            value_baseline="hidden_critic" if critic else "group_mean",
            state_critic_policy_hidden_projection_dim=2,
            state_critic_hidden_dim=8,
            state_critic_game_embedding_buckets=8,
            state_critic_game_embedding_dim=2,
            state_critic_epochs_per_window=2,
            state_critic_batch_size=1,
        )

        def tiny_model():
            # Plant the initial weights. Every spawned process seeds torch at
            # random, and about one draw in two hundred left the reject case's
            # lr=1.0 step inside the clip band on every probe token (seeds 226,
            # 228 and 324 do), so the window was accepted. From seed 0 each
            # token's log-ratio lands more than 0.9 nats outside the band.
            torch.manual_seed(0)
            model = _TinyVisionPolicy() if critic else _TinyPolicy()
            model.config = SimpleNamespace(hidden_size=2, use_cache=False)
            return model

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "training.ppo.AutoTokenizer.from_pretrained",
                lambda *_a, **_kw: SimpleNamespace(pad_token_id=0),
            )
            patch.setattr(
                FullWeightPPOTrainer, "_load_base_model", lambda *_a: tiny_model()
            )
            patch.setattr(
                FullWeightPPOTrainer,
                "_load_empty_policy_model",
                lambda *_a, **_kw: tiny_model().to("meta"),
            )
            patch.setattr(
                "training.multimodal.load_processing_stack", lambda _identity: object()
            )
            trainer = FullWeightPPOTrainer(
                config,
                grpo_group_size=None if critic else 2,
                grpo_puzzles_per_window=None if critic else 1,
                checkpoint_identity={
                    "environment": "arc3" if critic else "arc2",
                    "advantage_estimator": config.advantage_estimator,
                    "dataset_manifest": {"sha256": "dataset"},
                    # Only a harness that declared images pins a stack; the
                    # processor follows that pin, not the harness name.
                    **({} if critic else {"processing_stack": {"sha256": "p"}}),
                },
            )
        assert (trainer.state_critic is not None) is critic
        assert (trainer.processor is None) is critic
        initial = trainer._snapshot_update_state()
        rollouts = []
        for member, completion in enumerate(([3, 4], [4, 3])):
            sample = _weighted_sample()
            sample.input_ids = [1, 2, *completion]
            sample.labels = [-100, -100, *completion]
            batch = collate_samples([sample], pad_token_id=0)
            with torch.no_grad():
                scores = trainer._token_logprobs(batch)[0, -2:].tolist()
            rollouts.append(
                RolloutTrajectory(
                    trajectory_id=f"episode-{member}",
                    task_id="puzzle",
                    done=True,
                    trajectory_reward=float(member),
                    metadata={
                        "rollout_slot_id": f"slot-{member}",
                        "rollout_seed": 100 + member,
                        "behavior_policy_snapshot": {"iteration": -1},
                    },
                    steps=[
                        RolloutStep(
                            step_index=0,
                            prompt_messages=[],
                            tool_schemas=[],
                            assistant_message={},
                            reward=float(member),
                            done=True,
                            prompt_token_ids=[1, 2],
                            completion_token_ids=list(completion),
                            completion_logprobs=scores,
                        )
                    ],
                )
            )
        calls = 0

        def next_window_ready():
            nonlocal calls
            calls += 1
            # Polled before each extra cycle, plus once after the lazy
            # post-primary trust-region snapshot: two cycles need four polls.
            return calls >= 4

        random.seed(17)
        window_kwargs = dict(
            output_dir=Path(output_dir),
            iteration_index=0,
            policy_lag_windows=0,
            next_window_ready=next_window_ready,
            next_window_seconds_remaining=lambda: 600.0,
        )
        if critic:
            from training.full_weight_distributed import FullWeightDistributedRuntime

            runtime = FullWeightDistributedRuntime(control_group=dist.group.WORLD)
            runtime._trainer = trainer
            critic_before = copy.deepcopy(trainer.state_critic.state_dict())
            signals, _stats, signal_batch = trainer._compute_step_signals(rollouts)
            assert [item.return_to_go for item in signal_batch.observations] == [0, 1]
            assert all(
                len(item.policy_hidden) == 2 and any(item.policy_hidden)
                for item in signal_batch.observations
            )
            gathered = [None] * world_size
            dist.all_gather_object(gathered, signals)
            assert gathered[0] == gathered[1]
            # Actual runtime handler resets each rank to the same CPU RNG before
            # critic minibatch shuffling, even when their prior RNG states differ.
            torch.manual_seed(71 + rank)
            rng = [runtime._capture_rng_state() if rank == 0 else None]
            dist.broadcast_object_list(rng, src=0)
            result = runtime._handle_train_window(
                {
                    "rollouts": rollouts,
                    "rng_state": rng[0],
                    "output_dir": output_dir,
                    "iteration_index": 0,
                    "sample_budget_fraction": 1.0,
                    "policy_lag_windows": 0,
                    "entropy_coef": None,
                }
            )
        else:
            result = trainer.train_window(rollouts, **window_kwargs)
        from torch.distributed.tensor import DTensor

        # The final entropy probe must leave persistent shards ready for the
        # runtime's checkpoint and live policy publication.
        assert isinstance(trainer.model.model.embed_tokens.weight, DTensor)
        assert result.sample_count == 2
        assert result.baseline_update_steps == (4 if critic else 0)
        assert (
            result.adapter_dir
            is result.runtime_adapter_dir
            is result.trainer_state_path
            is None
        )
        assert result.window_update_rejected is reject
        if critic:
            assert result.optimizer_steps > 0
            assert result.baseline_mean_target == 0.5
            assert any(
                not torch.equal(trainer._local_tensor(parameter), backup)
                for parameter, backup in initial["parameters"]
            )
            saved = copy.deepcopy(trainer.state_critic.state_dict())
            assert saved["optimizer_state_dict"]["state"]
            assert any(
                not torch.equal(value, critic_before["network_state_dict"][name])
                for name, value in saved["network_state_dict"].items()
            )
            dist.all_gather_object(gathered, saved)
            _assert_exact(gathered[0], gathered[1])
            identity = {
                "iteration_index": 0,
                "model_name": config.model_name,
                "model_revision": config.model_revision,
                "checkpoint_identity": trainer._checkpoint_identity,
                "grpo_group_size": None,
                "grpo_puzzles_per_window": None,
            }
            checkpoint = runtime._handle_checkpoint(
                {
                    "root": str(Path(output_dir) / "checkpoints"),
                    "name": "critic",
                    "iteration_index": 0,
                    "metadata_fields": identity,
                }
            )
            with torch.no_grad():
                for parameter in trainer.state_critic.network.parameters():
                    parameter.add_(7)
                for state in trainer.state_critic.optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value):
                            value.add_(7)
            metadata = runtime._checkpointer_factory(checkpoint.parent).load(
                checkpoint, model=trainer.model, optimizer=trainer.optimizer
            )
            runtime._restore_checkpoint_payload(metadata, expected=identity)
            _assert_exact(trainer.state_critic.state_dict(), saved)
            return
        if reject:
            assert calls == result.extra_cycles == trainer.optimizer_step_count == 0
            assert result.extra_stop_reason == "window_update_rejected"
            # Every probe token left the band: the verdict is far from 0.1.
            assert result.post_update_clip_fraction == 1.0
            assert result.trust_region_lr_scale == 0.5
            assert not trainer.optimizer.state
            for parameter, backup in initial["parameters"]:
                torch.testing.assert_close(trainer._local_tensor(parameter), backup)
        else:
            assert result.rollout_logprob_parity_token_count == 4
            assert result.extra_cycles == 2
            assert result.extra_stop_reason == "window_ready"
            assert trainer.optimizer_step_count == 3
            assert any(
                not torch.equal(trainer._local_tensor(parameter), backup)
                for parameter, backup in initial["parameters"]
            )
        with pytest.raises(ValueError, match="Incomplete GRPO"):
            trainer.train_window(
                rollouts[:1], output_dir=Path(output_dir), iteration_index=1
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reject", [False, True])
def test_two_rank_complete_grpo_window_extra_cycles_and_rollback(
    tmp_path: Path, reject: bool
) -> None:
    spawn_ranks(
        _distributed_full_window_worker,
        2,
        str(tmp_path / "grpo_init"),
        str(tmp_path),
        reject,
    )


def test_two_rank_critic_window_hidden_features_and_checkpoint(tmp_path: Path) -> None:
    spawn_ranks(
        _distributed_full_window_worker,
        2,
        str(tmp_path / "critic_init"),
        str(tmp_path),
        False,
        True,
    )


def test_full_weight_loss_preserves_multimodal_evidence(monkeypatch) -> None:
    from test_multimodal_training import fixture_media

    processor, ids, payload, evidence = fixture_media()
    score = torch.nn.Parameter(torch.tensor(0.0))
    trainer = _loss_only_trainer(score)
    trainer.processor = processor
    sample = _weighted_sample()
    sample.input_ids = ids + [3, 4]
    sample.labels = [-100] * len(ids) + [3, 4]
    sample.attention_mask = [1] * len(sample.input_ids)
    sample.completion_mask = [0] * len(ids) + [1, 1]
    sample.prompt_token_count = len(ids)
    sample.old_token_logprobs = [0.0] * (len(sample.input_ids) - 1)
    sample.media = payload
    sample.media_evidence = evidence
    original = trainer._token_logprobs_and_entropies
    seen = []

    def token_scores(batch):
        seen.append(batch)
        assert batch["pixel_values"].shape == (32, 3)
        assert batch["position_ids"].shape == (3, 1, len(sample.input_ids))
        assert batch["mm_token_type_ids"].sum() == 8
        return original(batch)

    monkeypatch.setattr(trainer, "_token_logprobs_and_entropies", token_scores)
    trainer._micro_batch_forward_backward([sample], loss_denominator=1.0)
    assert len(seen) == 1
    assert score.grad.item() != 0.0
    sample.media_evidence = {
        **evidence,
        "image_hashes": list(reversed(evidence["image_hashes"])),
    }
    with pytest.raises(ValueError, match="server evidence"):
        trainer._micro_batch_forward_backward([sample], loss_denominator=1.0)
    assert len(seen) == 1


@pytest.mark.parametrize(
    "count,world_size,expected",
    [
        (8, 1, [(list(range(8)), [])]),
        (8, 3, [([0, 3, 6], []), ([1, 4, 7], []), ([2, 5], [2])]),
        (8, 4, [([0, 4], []), ([1, 5], []), ([2, 6], []), ([3, 7], [])]),
        (2, 4, [([0], []), ([1], []), ([], [0]), ([], [0])]),
        (0, 3, [([], []), ([], []), ([], [])]),
    ],
)
def test_padded_rank_shard_deals_round_robin_with_equal_forwards(
    count: int, world_size: int, expected
) -> None:
    items = list(range(count))
    shards = [
        padded_rank_shard(items, rank=rank, world_size=world_size)
        for rank in range(world_size)
    ]
    assert shards == expected
    assert sorted(item for share, _ in shards for item in share) == items
    assert {len(share) + len(padding) for share, padding in shards} == {
        math.ceil(count / world_size)
    }


@pytest.mark.parametrize("rank,world_size", [(3, 3), (-1, 3), (0, 0)])
def test_padded_rank_shard_rejects_ranks_outside_the_world(
    rank: int, world_size: int
) -> None:
    with pytest.raises(ValueError, match="outside a world"):
        padded_rank_shard([1, 2], rank=rank, world_size=world_size)


class _ThreadWorld:
    """In-process stand-in for torch.distributed: one thread per rank."""

    ReduceOp = dist.ReduceOp

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=30)
        self._local = threading.local()
        self._slots: list[torch.Tensor | None] = [None] * world_size

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rank(self) -> int:
        return self._local.rank

    def get_world_size(self) -> int:
        return self.world_size

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> None:
        self._slots[self.get_rank()] = tensor.detach().clone()
        self._barrier.wait()
        stacked = torch.stack(self._slots)
        if op == dist.ReduceOp.SUM:
            reduced = stacked.sum(dim=0)
        elif op == dist.ReduceOp.MAX:
            reduced = stacked.amax(dim=0)
        elif op == dist.ReduceOp.MIN:
            reduced = stacked.amin(dim=0)
        else:
            raise NotImplementedError(op)
        # Nobody overwrites a slot until every rank has read all of them.
        self._barrier.wait()
        tensor.copy_(reduced)

    def run(self, work):
        results: list = [None] * self.world_size
        errors: list = [None] * self.world_size

        def target(rank: int) -> None:
            self._local.rank = rank
            try:
                results[rank] = work(rank)
            except BaseException as exc:  # noqa: BLE001 - reported per rank
                # Peers stuck in a collective time out of the barrier; aborting
                # here could break peers that a reduction already released.
                errors[rank] = exc

        threads = [
            threading.Thread(target=target, args=(rank,))
            for rank in range(self.world_size)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads)
        return results, errors


def _probe_sample(index: int, *, completion_tokens: int) -> TokenizedSample:
    # The first token names the sample so fake forwards can identify it.
    prompt = [10 + index, 1]
    completion = [2] * completion_tokens
    input_ids = prompt + completion
    return TokenizedSample(
        trajectory_id=f"probe-{index}",
        step_index=0,
        input_ids=input_ids,
        labels=[-100] * len(prompt) + completion,
        attention_mask=[1] * len(input_ids),
        completion_mask=[0] * len(prompt) + [1] * completion_tokens,
        reward=0.0,
        return_to_go=0.0,
        advantage=1.0,
        prompt_token_count=len(prompt),
        old_token_logprobs=[0.0] * (len(input_ids) - 1),
        sequence_weight=1.0,
    )


def _token_span_entropy(batch, _logprobs, entropies):  # noqa: ANN001
    return {"span_entropy": entropies[batch["completion_mask"][:, 1:].bool()]}


def _sharded_probe_trainer(forwards: list[int]) -> FullWeightPPOTrainer:
    trainer = FullWeightPPOTrainer.__new__(FullWeightPPOTrainer)
    trainer.config = _training_config(max_window_clip_fraction=0.9, micro_batch_size=1)
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.updated = False

    def token_logprobs(batch):
        sample_index = batch["input_ids"][:, :1] - 10
        forwards.append(int(sample_index[0, 0]))
        # After the update odd samples move by e^0.5 > 1 + clip on every token.
        shift = 0.5 * (sample_index % 2) if trainer.updated else 0.0
        return torch.full(batch["input_ids"][:, 1:].shape, -1.0) + shift

    trainer._token_logprobs = token_logprobs
    return trainer


@pytest.mark.parametrize("count,world_size", [(8, 1), (8, 3), (8, 4), (2, 4)])
def test_sharded_trust_region_pools_token_counts_with_aligned_forwards(
    monkeypatch, caplog, count: int, world_size: int
) -> None:
    tokens = [1 + index % 3 for index in range(count)]
    samples = [
        _probe_sample(index, completion_tokens=tokens[index]) for index in range(count)
    ]
    total = sum(tokens)
    fraction = sum(tokens[1::2]) / total
    world = _ThreadWorld(world_size)
    monkeypatch.setattr("training.full_weight_ppo.dist", world)
    forwards: list[list[int]] = [[] for _ in range(world_size)]

    def rank_work(rank: int):
        trainer = _sharded_probe_trainer(forwards[rank])
        before = trainer._trust_region_token_logprobs(samples)
        pooled_tokens = trainer._trust_region_token_count(before)
        trainer.updated = True
        clip_fraction = trainer._post_update_clip_fraction(samples, before)
        rejected = trainer._apply_window_trust_region(
            {}, post_update_clip_fraction=clip_fraction, iteration_index=0
        )
        return before.numel(), pooled_tokens, clip_fraction, rejected

    with caplog.at_level(logging.INFO, logger="training.full_weight_ppo"):
        results, errors = world.run(rank_work)

    assert errors == [None] * world_size
    slots = math.ceil(count / world_size)
    for rank, (local_tokens, pooled_tokens, clip_fraction, rejected) in enumerate(
        results
    ):
        share = list(range(count))[rank::world_size]
        assert local_tokens == sum(tokens[index] for index in share)
        assert pooled_tokens == total
        assert clip_fraction == fraction
        assert not rejected
        # Before and after the update: the real share, then discarded padding.
        filler = share[0] if share else 0
        assert forwards[rank] == (share + [filler] * (slots - len(share))) * 2
    reports = [
        record.getMessage()
        for record in caplog.records
        if "post-update clip fraction" in record.getMessage()
    ]
    pooled_report = (
        f"window trust region post-update clip fraction={fraction:.6f} tokens={total}"
    )
    assert reports == [pooled_report] * world_size


@pytest.mark.parametrize(
    "lost_ranks,message",
    [({1}, "probe differs across ranks"), ({0, 1, 2}, "requires sampled probe")],
)
def test_sharded_trust_region_missing_scores_fail_on_every_rank(
    monkeypatch, lost_ranks: set[int], message: str
) -> None:
    samples = [_probe_sample(index, completion_tokens=1) for index in range(8)]
    world = _ThreadWorld(3)
    monkeypatch.setattr("training.full_weight_ppo.dist", world)

    def rank_work(rank: int) -> None:
        trainer = _sharded_probe_trainer([])
        before = trainer._trust_region_token_logprobs(samples)
        if rank in lost_ranks:
            trainer._trust_region_token_logprobs = lambda _samples: torch.empty(0)
        trainer._post_update_clip_fraction(samples, before)

    _, errors = world.run(rank_work)
    assert all(
        isinstance(error, RuntimeError) and message in str(error) for error in errors
    )


@pytest.mark.parametrize("world_size", [1, 3, 4])
def test_sharded_probe_entropy_pools_one_mean_with_aligned_forwards(
    monkeypatch, world_size: int
) -> None:
    tokens = [1 + index % 3 for index in range(8)]
    probe = [
        _probe_sample(index, completion_tokens=tokens[index]) for index in range(8)
    ]
    # Sample k carries entropy k + 1 on each of its span tokens.
    expected = sum((index + 1) * tokens[index] for index in range(8)) / sum(tokens)
    world = _ThreadWorld(world_size)
    monkeypatch.setattr("training.full_weight_ppo.dist", world)
    forwards: list[list[int]] = [[] for _ in range(world_size)]
    chunked: list[list[list[str]]] = [[] for _ in range(world_size)]

    def rank_work(rank: int) -> float:
        trainer = _sharded_probe_trainer(forwards[rank])
        trainer._probe_samples = probe

        def batch_iter(samples):
            chunked[rank].append([sample.trajectory_id for sample in samples])
            return iter([[sample] for sample in samples])

        def entropies(batch):
            sample_index = batch["input_ids"][:, :1] - 10
            forwards[rank].append(int(sample_index[0, 0]))
            values = (batch["input_ids"][:, 1:] * 0 + sample_index + 1).float()
            return torch.zeros_like(values), values

        trainer._batch_iter = batch_iter
        trainer._token_logprobs_and_entropies = entropies
        trainer._span_loss_terms = _token_span_entropy
        return trainer._compute_probe_entropy()

    results, errors = world.run(rank_work)

    assert errors == [None] * world_size
    assert results == [expected] * world_size
    slots = math.ceil(len(probe) / world_size)
    for rank in range(world_size):
        # Every rank chunks the whole probe, so micro-batch shuffling draws the
        # same global RNG values everywhere, then scores only its share.
        assert chunked[rank] == [[sample.trajectory_id for sample in probe]]
        share = list(range(8))[rank::world_size]
        assert forwards[rank] == share + [share[0]] * (slots - len(share))
