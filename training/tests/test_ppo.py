import json
import logging
from pathlib import Path

import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from training.config import LoraSettings, TrainingConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.baseline_batch import BaselineSignalBatch
from training.dataset import TokenizedSample
from training.rewards import AdvantageBatchStats
from training.ppo import (
    PPOTrainer,
    _assert_samples_fit_training_budgets,
    _avg_actions_per_completed_level,
    _extra_cycle_block_reason,
    _policy_lag_for_samples,
    _sample_budget_count,
    _filter_zero_signal_samples,
    _save_peft_adapter_without_model_card,
    _subsample_train_samples,
    _trainer_policy_iteration_at_window_start,
    summarize_sample_lengths,
    token_scores_from_hidden_states,
)
from training.state_critic import StateCriticBaseline
from training.runtime_lora import runtime_lora_has_effective_delta


class _DummyAdapterModel:
    def __init__(self) -> None:
        self.model_card_calls = 0
        self.save_calls = 0

    def create_or_update_model_card(self, _save_directory) -> None:
        self.model_card_calls += 1

    def save_pretrained(self, output_dir, safe_serialization: bool = True) -> None:
        assert safe_serialization is True
        self.save_calls += 1
        self.create_or_update_model_card(output_dir)
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")


class _MinimalRuntimeAdapterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj_lora_a = torch.nn.Parameter(
            torch.zeros((2, 2), dtype=torch.float32)
        )
        self.q_proj_lora_b = torch.nn.Parameter(
            torch.zeros((2, 2), dtype=torch.float32)
        )

    def create_or_update_model_card(self, _save_directory) -> None:
        return None

    def save_pretrained(self, output_dir, safe_serialization: bool = True) -> None:
        assert safe_serialization is True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Qwen3.5-9B",
                    "bias": "none",
                    "inference_mode": True,
                    "lora_alpha": 4,
                    "lora_dropout": 0.0,
                    "peft_type": "LORA",
                    "r": 2,
                    "target_modules": ["q_proj"],
                    "task_type": "CAUSAL_LM",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_file(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": self.q_proj_lora_a.detach().cpu(),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": self.q_proj_lora_b.detach().cpu(),
            },
            output_dir / "adapter_model.safetensors",
        )


class _FakePolicyModel(torch.nn.Module):
    def print_trainable_parameters(self) -> None:
        return None


class _DropoutPolicyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(2, 2)


class _TinyDecoder(torch.nn.Module):
    """Emits the shared logits parameter as per-position hidden states."""

    def __init__(self, logits: torch.nn.Parameter) -> None:
        super().__init__()
        self.logits = logits

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):  # noqa: ANN001
        del attention_mask
        batch_size, sequence_length = input_ids.shape
        hidden = self.logits.view(1, 1, -1).expand(batch_size, sequence_length, -1)
        return type("_TinyDecoderOutput", (), {"last_hidden_state": hidden})()


class _TinyPolicy(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 5) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self.model = _TinyDecoder(self.logits)
        self.lm_head = torch.nn.Identity()

    def get_base_model(self) -> "_TinyPolicy":
        return self


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    return load_file(adapter_dir / "adapter_model.safetensors")


def _tensor_dicts_match(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_policy_logprob_mode_keeps_model_training_while_disabling_dropout() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = type("_Config", (), {"disable_dropout": True})()
    trainer.model = _DropoutPolicyModel()
    trainer.model.eval()

    trainer._set_policy_mode_for_logprob_computation()

    assert trainer.model.training is True
    assert trainer.model.linear.training is True
    assert trainer.model.dropout.training is False


def test_policy_logprob_mode_leaves_dropout_training_when_enabled() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = type("_Config", (), {"disable_dropout": False})()
    trainer.model = _DropoutPolicyModel()
    trainer.model.eval()

    trainer._set_policy_mode_for_logprob_computation()

    assert trainer.model.training is True
    assert trainer.model.linear.training is True
    assert trainer.model.dropout.training is True


def test_assert_samples_fit_training_budgets_rejects_oversized_samples() -> None:
    short = TokenizedSample(
        trajectory_id="short",
        step_index=0,
        input_ids=[1, 2, 3],
        labels=[-100, -100, 3],
        attention_mask=[1, 1, 1],
        completion_mask=[0, 0, 1],
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
    )
    long = TokenizedSample(
        trajectory_id="long",
        step_index=0,
        input_ids=[1, 2, 3, 4],
        labels=[-100, -100, -100, 4],
        attention_mask=[1, 1, 1, 1],
        completion_mask=[0, 0, 0, 1],
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
    )

    with pytest.raises(ValueError, match="must fit the configured token budget"):
        _assert_samples_fit_training_budgets(
            [short, long],
            max_sequence_tokens=3,
        )


def _make_runtime_export_test_trainer(
    monkeypatch, *, learning_rate: float
) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        value_baseline="hidden_critic",
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=learning_rate,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = type("_Tokenizer", (), {"pad_token_id": 0})()
    trainer.model = _MinimalRuntimeAdapterModel()
    trainer.optimizer = torch.optim.AdamW(
        trainer._trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    trainer._optimizer_step_count = 0
    trainer._trainable_parameter_count = sum(
        parameter.numel() for parameter in trainer._trainable_parameters()
    )
    trainer.state_critic = _test_state_critic()
    trainer._extract_policy_hidden_states = lambda _rollouts: {}

    sample = TokenizedSample(
        trajectory_id="traj-0",
        policy_iteration=0,
        step_index=0,
        input_ids=[1, 2],
        labels=[-100, 2],
        attention_mask=[1, 1],
        completion_mask=[0, 1],
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
    )

    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])

    def fake_attach_logprobs(samples: list[TokenizedSample]) -> None:
        for item in samples:
            item.old_token_logprobs = [0.0] * max(0, len(item.input_ids) - 1)
            item.old_logprob = float(sum(item.old_token_logprobs))

    def fake_token_logprobs(batch: dict[str, torch.Tensor], **_kwargs) -> torch.Tensor:
        token_count = max(0, batch["input_ids"].shape[1] - 1)
        signal = trainer.model.q_proj_lora_b.mean()
        return signal.expand(batch["input_ids"].shape[0], token_count)

    monkeypatch.setattr(trainer, "_attach_logprobs", fake_attach_logprobs)
    monkeypatch.setattr(trainer, "_token_logprobs", fake_token_logprobs)
    monkeypatch.setattr(
        trainer,
        "_token_logprobs_and_entropies",
        lambda batch, **_kwargs: (
            (logprobs := fake_token_logprobs(batch)),
            torch.zeros_like(logprobs),
        ),
    )
    return trainer


def _dummy_rollout() -> RolloutTrajectory:
    return RolloutTrajectory(
        trajectory_id="traj-0",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=False,
            ),
            RolloutStep(
                step_index=3,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
            ),
        ],
        trajectory_reward=1.0,
        metadata={"run_total_actions": 1},
    )


def _single_step_rollout() -> RolloutTrajectory:
    return RolloutTrajectory(
        trajectory_id="tiny-traj",
        task_id="tiny-task",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=True,
            )
        ],
        trajectory_reward=1.0,
        metadata={"run_total_actions": 1},
    )


def _make_tiny_ppo_trainer(
    *, advantage: float, learning_rate: float = 0.25
) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        value_baseline="hidden_critic",
        max_window_clip_fraction=0.0,
        model_name="tiny",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=8,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=learning_rate,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=100.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = type("_Tokenizer", (), {"pad_token_id": 0})()
    trainer.model = _TinyPolicy()
    trainer.optimizer = torch.optim.AdamW(
        trainer._trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    trainer._optimizer_step_count = 0
    trainer._trainable_parameter_count = sum(
        parameter.numel() for parameter in trainer._trainable_parameters()
    )
    trainer._cuda_trainer_model_allocated_gb = 0.0
    trainer.state_critic = _test_state_critic()
    trainer._extract_policy_hidden_states = lambda _rollouts: {}
    return trainer


def _tiny_action_sample(
    *, advantage: float, action_token_id: int = 2
) -> TokenizedSample:
    initial_action_logprob = float(
        torch.log_softmax(torch.zeros(5, dtype=torch.float32), dim=-1)[action_token_id]
    )
    return TokenizedSample(
        trajectory_id="tiny-traj",
        step_index=0,
        input_ids=[1, action_token_id],
        labels=[-100, action_token_id],
        attention_mask=[1, 1],
        completion_mask=[0, 1],
        reward=advantage,
        return_to_go=advantage,
        advantage=advantage,
        old_logprob=initial_action_logprob,
        old_token_logprobs=[initial_action_logprob],
    )


def _tiny_action_logprob(trainer: PPOTrainer, *, action_token_id: int = 2) -> float:
    with torch.no_grad():
        return float(
            torch.log_softmax(trainer.model.logits.detach(), dim=-1)[
                action_token_id
            ].item()
        )


def _run_tiny_ppo_step(
    monkeypatch, tmp_path: Path, *, advantage: float
) -> tuple[float, float]:
    trainer = _make_tiny_ppo_trainer(advantage=advantage)
    sample = _tiny_action_sample(advantage=advantage)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    before = _tiny_action_logprob(trainer)
    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )
    after = _tiny_action_logprob(trainer)

    assert result.sample_count == 1
    assert result.optimizer_steps == 1
    assert result.mean_ratio == pytest.approx(1.0)
    return before, after


def test_tiny_ppo_step_positive_advantage_increases_action_logprob(
    monkeypatch, tmp_path: Path
) -> None:
    before, after = _run_tiny_ppo_step(monkeypatch, tmp_path, advantage=1.0)

    assert after > before


def test_tiny_ppo_step_negative_advantage_decreases_action_logprob(
    monkeypatch, tmp_path: Path
) -> None:
    before, after = _run_tiny_ppo_step(monkeypatch, tmp_path, advantage=-1.0)

    assert after < before


@pytest.mark.parametrize("policy_loss", ["ppo", "dppo_binary_tv"])
def test_group_mean_window_trains_25_original_rollouts_in_16_and_9_minibatches(
    monkeypatch,
    tmp_path: Path,
    policy_loss: str,
) -> None:
    from dataclasses import replace

    trainer = _make_tiny_ppo_trainer(advantage=1, learning_rate=1e-3)
    trainer.config.policy_loss = policy_loss
    trainer.config.value_baseline = "group_mean"
    trainer.config.loss_aggregation = "sequence_sum_fixed"
    trainer.config.loss_normalization_tokens = 32
    trainer.config.optimizer_minibatch_rollouts = 16
    trainer.config.episode_packing = True
    trainer.config.previous_policy_kl_coef = 0.02 if policy_loss == "ppo" else 0
    trainer.config.max_window_clip_fraction = 0.1 if policy_loss == "ppo" else 0
    trainer.state_critic = None
    rollouts = []
    for index in range(25):
        score = 0.0 if index < 12 else (100.0 if index < 24 else 50.0)
        step = RolloutStep(
            step_index=0,
            prompt_messages=[],
            tool_schemas=[],
            assistant_message={},
            reward=score,
            done=True,
            prompt_token_ids=[1],
            completion_token_ids=[2],
            completion_logprobs=[-math.log(5)],
        )
        steps = [] if index == 3 else [step]
        if index == 0:
            steps.append(
                replace(
                    step,
                    step_index=1,
                    reward=0,
                    prompt_token_ids=[3],
                    completion_token_ids=[4],
                    metadata={"segment_start": True},
                )
            )
        rollouts.append(
            RolloutTrajectory(
                trajectory_id=f"episode-{index}",
                task_id="ar25",
                steps=steps,
                trajectory_reward=score,
                metadata={
                    "final_arc_score": score,
                    "start_level": 0,
                    "max_runtime_seconds": 30.0,
                    "policy_iteration": -1,
                },
            )
        )
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "runtime",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **kw: output_dir / "state.pt",
    )
    cycles = []
    execute = trainer._execute_train_cycle

    def record_cycle(cycle, **kwargs):
        cycles.append(
            (
                [sample.trajectory_id for chunk in cycle for sample in chunk],
                kwargs["loss_denominator"],
            )
        )
        return execute(cycle, **kwargs)

    monkeypatch.setattr(trainer, "_execute_train_cycle", record_cycle)
    result = trainer.train_window(rollouts, output_dir=tmp_path, iteration_index=0)
    assert result.optimizer_steps == 2
    assert [denominator for _ids, denominator in cycles] == [16 * 32, 9 * 32]
    assert cycles[0][0].count("episode-0") == 2
    assert (
        "episode-24" in cycles[1][0]
    )  # Zero advantage retains denominator and forward.
    assert result.filtered_zero_signal_sample_count == 0
    assert result.policy_update_metrics["all/tokens"] == 25
    assert result.baseline_update_steps == 0
    assert result.baseline_metrics["group_count"] == 1
    assert trainer.state_critic is None


def _tiny_entropy_control_trainer() -> PPOTrainer:
    from training.config import EntropyControlConfig

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.entropy_control = EntropyControlConfig(enabled=True)
    return trainer


def test_probe_states_capture_persist_and_reload(tmp_path: Path) -> None:
    trainer = _tiny_entropy_control_trainer()
    samples = [_tiny_action_sample(advantage=1.0) for _ in range(10)]
    for index, sample in enumerate(samples):
        sample.trajectory_id = f"traj-{index}"

    trainer._ensure_probe_states(samples, tmp_path)

    probe_path = tmp_path / "probe_states.pt"
    assert probe_path.exists()
    assert [sample.trajectory_id for sample in trainer._probe_samples] == [
        f"traj-{index}" for index in range(8)
    ]
    # The probe is a deep copy: mutating the live batch must not touch it.
    samples[0].advantage = -5.0
    assert trainer._probe_samples[0].advantage == 1.0

    # A fresh trainer (construction/resume) reloads the persisted probe
    # instead of re-capturing from the new window's samples.
    resumed = _tiny_entropy_control_trainer()
    replacement = [_tiny_action_sample(advantage=1.0)]
    replacement[0].trajectory_id = "different"
    resumed._ensure_probe_states(replacement, tmp_path)
    assert [sample.trajectory_id for sample in resumed._probe_samples] == [
        f"traj-{index}" for index in range(8)
    ]

    # Guard revision: a corrupt probe file is re-captured, not fatal.
    probe_path.write_bytes(b"garbage")
    recaptured = _tiny_entropy_control_trainer()
    recaptured._ensure_probe_states(replacement, tmp_path)
    assert [sample.trajectory_id for sample in recaptured._probe_samples] == [
        "different"
    ]
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    assert [sample.trajectory_id for sample in payload["samples"]] == ["different"]


def test_probe_states_active_when_entropy_control_disabled(tmp_path: Path) -> None:
    # The probe is pure observability: the composition-stable entropy series
    # is a collapse signal in its own right (in either direction), so it runs
    # regardless of whether the entropy controller is steering.
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    assert trainer.config.entropy_control.enabled is False

    trainer._ensure_probe_states([_tiny_action_sample(advantage=1.0)], tmp_path)

    assert trainer._probe_samples
    assert (tmp_path / "probe_states.pt").exists()
    # The tiny 5-token softmax policy has strictly positive entropy.
    assert trainer._compute_probe_entropy() > 0.0


def test_train_window_reports_entropy_even_when_entropy_bonus_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    assert trainer.config.entropy_coef == 0.0
    assert trainer.config.entropy_control.enabled is False
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )

    assert result.mean_entropy > 0.0
    assert result.entropy_loss == pytest.approx(0.0)
    assert result.entropy_coef_used == pytest.approx(0.0)


def test_train_window_reports_probe_entropy_when_entropy_control_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _tiny_entropy_control_trainer()
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )

    assert (tmp_path / "probe_states.pt").exists()
    # The tiny 5-token softmax policy has strictly positive entropy.
    assert result.probe_entropy > 0.0
    # entropy_coef is 0.0 here, so the grad-ratio probe stays gated off.
    assert result.entropy_to_policy_grad_ratio == 0.0


def test_entropy_to_policy_grad_ratio_measured_on_tiny_policy() -> None:
    trainer = _tiny_entropy_control_trainer()
    trainer.live_entropy_coef = 0.01
    # Non-uniform logits: at the uniform initialization the entropy gradient
    # is exactly zero (entropy is maximized), which would mask the signal.
    with torch.no_grad():
        trainer.model.logits.copy_(torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0]))
    sample = _tiny_action_sample(advantage=1.0)

    ratio = trainer._entropy_to_policy_grad_ratio([sample])

    assert ratio > 0.0
    # Optimizer state and param.grad must be untouched by the probe.
    assert all(parameter.grad is None for parameter in trainer._trainable_parameters())

    # Guards: live coefficient 0 and empty chunks report 0.0.
    trainer.live_entropy_coef = 0.0
    assert trainer._entropy_to_policy_grad_ratio([sample]) == 0.0
    trainer.live_entropy_coef = 0.01
    assert trainer._entropy_to_policy_grad_ratio([]) == 0.0
    # A fixed entropy coefficient is measured even without adaptive control.
    disabled = _make_tiny_ppo_trainer(advantage=1.0)
    disabled.live_entropy_coef = 0.01
    with torch.no_grad():
        disabled.model.logits.copy_(torch.tensor([1.0, 0.5, 0.0, -0.5, -1.0]))
    assert disabled._entropy_to_policy_grad_ratio([sample]) > 0.0


def test_rollout_logprob_parity_accepts_small_numeric_drift() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)

    failure = trainer._rollout_logprob_parity_failure(
        {
            "token_count": 100,
            "mean_abs": 0.001,
            "p95_abs": 0.002,
            "max_abs": 0.01,
            "signed_mean": 0.0,
        },
        sample_count=1,
    )

    assert failure == ""


def test_rollout_logprob_parity_accepts_rare_outlier() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)

    failure = trainer._rollout_logprob_parity_failure(
        {
            "token_count": 3412,
            "mean_abs": 0.00982949,
            "p95_abs": 0.0678051,
            "max_abs": 2.15615,
            "outlier_count": 1,
            "outlier_fraction": 1 / 3412,
            "signed_mean": -0.00129848,
        },
        sample_count=6,
    )

    assert failure == ""


def test_trainer_policy_iteration_lifecycle_at_window_start() -> None:
    # Iteration N-1 exports adapter_iter_{N-1} at its END, before iteration
    # N's train_window is entered, so at the start of iteration N the trainer
    # already holds adapter_iter_{N-1}. Only iteration 0 runs on the base
    # policy (-1).
    assert _trainer_policy_iteration_at_window_start(0) == -1  # base policy
    assert _trainer_policy_iteration_at_window_start(1) == 0  # adapter 0
    assert _trainer_policy_iteration_at_window_start(2) == 1  # adapter 1
    assert _trainer_policy_iteration_at_window_start(178) == 177


def test_policy_lag_for_samples_requires_one_coherent_not_future_policy() -> None:
    samples = [_tiny_action_sample(advantage=1.0)]
    samples[0].policy_iteration = 4
    assert _policy_lag_for_samples(samples, trainer_policy_iteration=5) == 1

    mixed = [_tiny_action_sample(advantage=1.0) for _ in range(2)]
    mixed[0].policy_iteration = 4
    mixed[1].policy_iteration = 5
    with pytest.raises(RuntimeError, match="mixed rollout policy"):
        _policy_lag_for_samples(mixed, trainer_policy_iteration=5)

    with pytest.raises(RuntimeError, match="newer than the trainer"):
        _policy_lag_for_samples(samples, trainer_policy_iteration=3)


def test_rollout_logprob_parity_tolerates_one_deep_tail_token() -> None:
    """One token many nats off among ~20k, with mean and p95 tight, is tail
    noise of T=1 sampling scored in bf16, not a systematic mismatch: the
    2026-09-02 control arm died at boot on exactly this (max 8.54 nats,
    mean 0.011, 1 outlier in 20,976 tokens)."""
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    metrics = {
        "token_count": 19638,
        "mean_abs": 0.0152,
        "p95_abs": 0.0875,
        "max_abs": 12.87,
        "outlier_count": 1,
        "outlier_fraction": 1 / 19638,
        "signed_mean": -0.0016,
    }

    assert (
        trainer._rollout_logprob_parity_failure(
            {**metrics, "checked_policy_lag": 0}, sample_count=6
        )
        == ""
    )
    with pytest.raises(RuntimeError, match="only valid"):
        trainer._rollout_logprob_parity_failure(
            {**metrics, "checked_policy_lag": 1}, sample_count=6
        )


def test_rollout_logprob_parity_tolerance_set() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    healthy = {
        "token_count": 19524,
        "mean_abs": 0.0110,
        "p95_abs": 0.0751,
        "max_abs": 1.35,
        "outlier_count": 0,
        "outlier_fraction": 0.0,
        "signed_mean": 0.0,
        "checked_policy_lag": 0,
    }
    assert trainer._rollout_logprob_parity_failure(healthy, sample_count=6) == ""

    # A recompute-pipeline fault (mean 0.0358, 37 outliers, max 12.49) is a
    # FAULT under the gates.
    faulty = {
        **healthy,
        "mean_abs": 0.0358,
        "p95_abs": 0.161,
        "max_abs": 12.49,
        "outlier_count": 37,
        "outlier_fraction": 37 / 19192,
        "token_count": 19192,
    }
    assert trainer._rollout_logprob_parity_failure(faulty, sample_count=6)


def test_parity_check_counts_completion_tokens() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="stub",
        output_dir="unused",
        device="cpu",
        seed=0,
        max_sequence_tokens=4096,
        micro_batch_size=1,
        max_micro_batch_tokens=4096,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = type("_Tokenizer", (), {"pad_token_id": 0})()
    trainer._set_policy_mode_for_training_forward = lambda: None  # type: ignore[method-assign]
    trainer._token_logprobs = lambda batch, **_kwargs: batch[  # type: ignore[method-assign]
        "old_token_logprobs"
    ].clone()

    def sample(name: str) -> TokenizedSample:
        return TokenizedSample(
            trajectory_id=name,
            step_index=0,
            input_ids=[1, 2, 3, 4],
            labels=[-100, -100, 3, 4],
            attention_mask=[1, 1, 1, 1],
            completion_mask=[0, 0, 1, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=1.0,
            old_token_logprobs=[-0.5, -0.5, -0.5],
        )

    result = trainer._check_rollout_logprob_parity([sample("a"), sample("b")])

    assert result["token_count"] == 4


def test_rollout_logprob_parity_rejects_bad_old_policy_ratios() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)

    failure = trainer._rollout_logprob_parity_failure(
        {
            "token_count": 100,
            "mean_abs": 0.25,
            "p95_abs": 1.0,
            "max_abs": 4.0,
            "outlier_count": 10,
            "outlier_fraction": 0.1,
            "signed_mean": 0.0,
        },
        sample_count=1,
    )

    assert "outside tolerances" in failure


def test_non_quantized_cuda_policy_load_uses_bf16_dtype(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cuda:1",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=True,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cuda:1")
    trainer.device_index = 1

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):  # noqa: ANN001
            captured_kwargs.update(kwargs)
            assert model_name == "Qwen/Qwen3.5-9B"
            return _FakePolicyModel()

    monkeypatch.setattr("training.ppo.AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(
        "training.ppo.get_peft_model",
        lambda _base_model, _lora_config: _FakePolicyModel(),
    )

    trainer._load_policy_model()

    assert captured_kwargs["torch_dtype"] is torch.bfloat16
    assert "quantization_config" not in captured_kwargs
    assert captured_kwargs["attn_implementation"] == "sdpa"


def test_explicit_flash_attention_policy_load_passes_attn_implementation(
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, object] = {}
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cuda:1",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=True,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
        attn_implementation="flash_attention_2",
    )
    trainer.device = torch.device("cuda:1")
    trainer.device_index = 1

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):  # noqa: ANN001
            captured_kwargs.update(kwargs)
            assert model_name == "Qwen/Qwen3.5-9B"
            return _FakePolicyModel()

    monkeypatch.setattr("training.ppo.AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(
        "training.ppo.get_peft_model",
        lambda _base_model, _lora_config: _FakePolicyModel(),
    )

    trainer._load_policy_model()

    assert captured_kwargs["attn_implementation"] == "flash_attention_2"


def test_resolved_fa4_backend_is_explicit_and_available(monkeypatch) -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = type("Config", (), {"attn_implementation": "flash_attention_4"})()
    monkeypatch.setattr("training.ppo.require_fa4_distribution", lambda: "4.0.0b19")

    assert trainer._resolved_attn_implementation() == "flash_attention_4"


def test_resolved_fa4_backend_requires_cute_package(monkeypatch) -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = type("Config", (), {"attn_implementation": "flash_attention_4"})()

    def missing_distribution() -> str:
        raise RuntimeError("flash-attn-4 is not installed")

    monkeypatch.setattr("training.ppo.require_fa4_distribution", missing_distribution)

    with pytest.raises(RuntimeError, match="requires the locked flash-attn-4 package"):
        trainer._resolved_attn_implementation()


def test_auto_attention_policy_load_is_rejected() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cuda:1",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=True,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
        attn_implementation="auto",
    )

    with pytest.raises(ValueError, match="attn_implementation must be explicit"):
        trainer._resolved_attn_implementation()


def test_policy_load_enables_checkpointing_and_disables_dropout(monkeypatch) -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=True,
        lora=LoraSettings(r=2, alpha=4, dropout=0.25, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.device_index = None

    class _FakeCheckpointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("_Config", (), {"use_cache": True})()
            self.dropout = torch.nn.Dropout(p=0.25)
            self.checkpointing_kwargs = None
            self.input_require_grads_enabled = False
            self.is_gradient_checkpointing = False

        def gradient_checkpointing_enable(
            self, *, gradient_checkpointing_kwargs
        ) -> None:
            self.checkpointing_kwargs = gradient_checkpointing_kwargs
            self.is_gradient_checkpointing = True

        def enable_input_require_grads(self) -> None:
            self.input_require_grads_enabled = True

        def print_trainable_parameters(self) -> None:
            return None

    fake_model = _FakeCheckpointModel()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):  # noqa: ANN001
            del kwargs
            assert model_name == "Qwen/Qwen3.5-9B"
            return fake_model

    def _fake_get_peft_model(base_model, lora_config):  # noqa: ANN001
        assert base_model is fake_model
        assert lora_config.lora_dropout == pytest.approx(0.0)
        return base_model

    monkeypatch.setattr("training.ppo.AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr("training.ppo.get_peft_model", _fake_get_peft_model)

    model = trainer._load_policy_model()

    assert model is fake_model
    assert fake_model.config.use_cache is False
    # disable_dropout=True makes the recomputed forward deterministic, so the
    # trainer skips the per-checkpoint RNG-state stash.
    assert fake_model.checkpointing_kwargs == {
        "use_reentrant": False,
        "preserve_rng_state": False,
    }
    assert fake_model.input_require_grads_enabled is True
    assert fake_model.dropout.p == pytest.approx(0.0)


def test_training_forward_mode_keeps_model_training_with_dropout_disabled() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.model.eval()

    trainer._set_policy_mode_for_training_forward()

    assert trainer.model.training is True


def test_train_window_uses_training_forward_mode_for_optimizer(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    mode_during_logprob_calls: list[bool] = []
    original_token_logprobs = trainer._token_logprobs

    def wrapped_token_logprobs(
        batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        mode_during_logprob_calls.append(bool(trainer.model.training))
        return original_token_logprobs(batch, **kwargs)

    monkeypatch.setattr(trainer, "_token_logprobs", wrapped_token_logprobs)

    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )

    assert result.optimizer_steps == 1
    assert mode_during_logprob_calls[-1] is True


def _stub_signal_batch() -> BaselineSignalBatch:
    """A critic-free window batch for tests that stub the signal computation."""
    return BaselineSignalBatch(
        per_rollout_signals=[],
        stats=AdvantageBatchStats(
            rollout_count=0,
            mean_reward=0.0,
            std_reward=0.0,
            min_reward=0.0,
            max_reward=0.0,
            mean_advantage=0.0,
            std_advantage=0.0,
        ),
        rollouts=[],
        mean_value=0.0,
        mean_target=0.0,
        mean_raw_advantage=0.0,
        std_raw_advantage=0.0,
        explained_variance=0.0,
        diagnostics={"mean_absolute_error": 0.0},
    )


@pytest.mark.parametrize("world_size,fail_after", [(1, 1), (1, 2), (2, 1)])
def test_fatal_health_during_accumulation_prevents_step_and_export(
    monkeypatch, tmp_path: Path, world_size: int, fail_after: int
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.advantage_estimator = "grpo"
    trainer.config.gradient_accumulation_steps = 2
    trainer.state_critic = None
    stub_batch = _stub_signal_batch()
    monkeypatch.setattr(
        trainer, "_compute_step_signals", lambda _: ({}, stub_batch.stats, stub_batch)
    )
    trainer._remember_parity_policy(-1)
    samples = [_tiny_action_sample(advantage=1.0) for _ in range(2 * world_size)]
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: samples)
    for method in ("_export_policy_snapshot", "_save_training_state"):
        monkeypatch.setattr(
            trainer, method, lambda *args, **kwargs: pytest.fail("export after failure")
        )
    initial_weights = [p.detach().clone() for p in trainer._trainable_parameters()]
    events = []
    completed_microbatches = 0
    forward_backward = trainer._micro_batch_forward_backward

    def fail_during_accumulation(*args, **kwargs):
        nonlocal completed_microbatches
        metrics = forward_backward(*args, **kwargs)
        completed_microbatches += 1
        events.append("microbatch")
        return metrics

    def check_health():
        if completed_microbatches >= fail_after:
            events.append("fatal")
            raise RuntimeError("evaluation failed during accumulation")

    class FakeDdpGroup:
        world_size = 2

        def send_cycle(self, shares, **kwargs):
            assert len(shares) == 1 and len(shares[0]) == 2
            events.append("dispatch")

        def reduce_grads_into_params(self):
            events.append("reduce")

        def broadcast_weights(self):
            for before, after in zip(initial_weights, trainer._trainable_parameters()):
                assert torch.equal(before, after)
            events.append("unchanged_weights")

        def gather_worker_metrics(self):
            events.append("gather")
            return []

    if world_size > 1:
        trainer._ddp_group = FakeDdpGroup()
    monkeypatch.setattr(
        trainer, "_micro_batch_forward_backward", fail_during_accumulation
    )
    with pytest.raises(RuntimeError, match="evaluation failed during accumulation"):
        trainer.train_window(
            [_single_step_rollout()],
            output_dir=tmp_path,
            iteration_index=1,
            policy_lag_windows=1,
            check_health=check_health,
        )

    assert trainer.optimizer_step_count == 0
    assert not trainer.optimizer.state
    assert trainer.state_critic is None
    for before, after in zip(initial_weights, trainer._trainable_parameters()):
        assert torch.equal(before, after)
    if world_size == 1:
        assert completed_microbatches == fail_after
        assert events[-1] == "fatal"
    else:
        assert events == [
            "dispatch",
            "microbatch",
            "microbatch",
            "reduce",
            "gather",
            "fatal",
            "unchanged_weights",
            "gather",
        ]


@pytest.mark.parametrize("advantage", [0.0, 1.0])
def test_fatal_health_before_export_blocks_zero_signal_and_updated_windows(
    monkeypatch, tmp_path: Path, advantage: float
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=advantage)
    trainer.config.advantage_estimator = "grpo"
    trainer.state_critic = None
    stub_batch = _stub_signal_batch()
    monkeypatch.setattr(
        trainer, "_compute_step_signals", lambda _: ({}, stub_batch.stats, stub_batch)
    )
    trainer._remember_parity_policy(-1)
    sample = _tiny_action_sample(advantage=advantage)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    failed = False

    def latch_failure(*args):
        nonlocal failed
        failed = True
        return 0.0

    # Both paths have work between entry and export: sample preparation for
    # a zero-signal window, and the final probe forward after a real update.
    monkeypatch.setattr(
        trainer,
        "_ensure_probe_states" if advantage == 0.0 else "_compute_probe_entropy",
        latch_failure,
    )
    for method in ("_export_policy_snapshot", "_save_training_state"):
        monkeypatch.setattr(
            trainer, method, lambda *args, **kwargs: pytest.fail("export after failure")
        )

    def check_health():
        if failed:
            raise RuntimeError("evaluation failed before export")

    with pytest.raises(RuntimeError, match="evaluation failed before export"):
        trainer.train_window(
            [_single_step_rollout()],
            output_dir=tmp_path,
            iteration_index=1,
            policy_lag_windows=1,
            check_health=check_health,
        )
    assert trainer.optimizer_step_count == int(advantage != 0.0)


def test_train_window_runs_only_deadline_safe_extra_cycles(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0, learning_rate=1e-5)
    trainer.config.extra_train_cycles_until_window = True
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )
    eta_calls = {"count": 0}

    def seconds_remaining() -> float:
        eta_calls["count"] += 1
        return 1000.0 if eta_calls["count"] == 1 else 0.0

    result = trainer.train_window(
        [_single_step_rollout()],
        output_dir=tmp_path,
        iteration_index=0,
        next_window_ready=lambda: False,
        next_window_seconds_remaining=seconds_remaining,
    )

    assert result.processed_tokens == 2 * len(sample.input_ids)
    assert result.forward_time_s > 0
    assert result.backward_time_s > 0
    assert result.optimizer_cycle_seconds > result.forward_time_s + result.backward_time_s
    assert result.extra_cycles == 1
    assert result.extra_optimizer_steps == 1
    assert result.optimizer_steps == 2
    assert result.extra_stop_reason == "insufficient_rollout_slack"


def test_train_window_rejects_extra_cycles_below_full_coverage(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0, learning_rate=1e-5)
    trainer.config.extra_train_cycles_until_window = True
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    result = trainer.train_window(
        [_single_step_rollout()],
        output_dir=tmp_path,
        iteration_index=0,
        sample_budget_fraction=0.99,
        next_window_ready=lambda: False,
        next_window_seconds_remaining=lambda: 1000.0,
    )

    assert result.extra_cycles == 0
    assert result.optimizer_steps == 1
    assert result.extra_stop_reason == "not_full_coverage"


def _failing_parity_metrics() -> dict[str, float | int]:
    # Healthy distribution stats with a single fat-tail outlier token: the
    # stochastic vLLM-vs-trainer divergence observed on live 27B windows.
    return {
        "token_count": 200,
        "mean_abs": 0.01,
        "p95_abs": 0.05,
        "max_abs": 3.0,
        "outlier_count": 1,
        "outlier_fraction": 0.005,
        "signed_mean": 0.0,
    }


def _parity_test_trainer(monkeypatch) -> PPOTrainer:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )
    return trainer


def test_same_policy_parity_failure_is_immediately_fatal(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _parity_test_trainer(monkeypatch)
    monkeypatch.setattr(
        trainer,
        "_check_rollout_logprob_parity",
        lambda samples, **kwargs: _failing_parity_metrics(),
    )

    with pytest.raises(
        RuntimeError, match="Same-checkpoint logprob parity check failed"
    ):
        trainer.train_window(
            [_single_step_rollout()],
            output_dir=tmp_path,
            iteration_index=0,
            policy_lag_windows=0,
        )
    assert trainer._optimizer_step_count == 0


def test_lag_one_window_requires_cached_sampler_weights(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _parity_test_trainer(monkeypatch)
    with pytest.raises(RuntimeError, match="no cached weights for sampler policy"):
        trainer.train_window(
            [_single_step_rollout()],
            output_dir=tmp_path,
            iteration_index=1,
            policy_lag_windows=1,
        )
    assert trainer._optimizer_step_count == 0


def test_rollout_logprob_parity_uses_training_forward_mode() -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=True,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = type("_Tokenizer", (), {"pad_token_id": 0})()
    mode_calls: list[str] = []

    trainer._set_policy_mode_for_logprob_computation = (  # type: ignore[method-assign]
        lambda: mode_calls.append("logprob")
    )
    trainer._set_policy_mode_for_training_forward = (  # type: ignore[method-assign]
        lambda: mode_calls.append("training")
    )
    trainer._token_logprobs = lambda batch, **_kwargs: batch[  # type: ignore[method-assign]
        "old_token_logprobs"
    ].clone()

    sample = TokenizedSample(
        trajectory_id="traj",
        step_index=0,
        input_ids=[1, 2, 3],
        labels=[-100, -100, 3],
        attention_mask=[1, 1, 1],
        completion_mask=[0, 0, 1],
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
        old_token_logprobs=[0.0, -0.5],
    )

    result = trainer._check_rollout_logprob_parity([sample])

    assert mode_calls == ["training"]
    assert result["token_count"] == 1
    assert result["mean_abs"] == pytest.approx(0.0)


def test_save_peft_adapter_without_model_card_disables_template_generation(
    tmp_path,
) -> None:
    model = _DummyAdapterModel()

    _save_peft_adapter_without_model_card(model, tmp_path)

    assert model.save_calls == 1
    assert model.model_card_calls == 0
    assert (tmp_path / "adapter_config.json").exists()

    model.create_or_update_model_card(tmp_path)
    assert model.model_card_calls == 1


def test_token_scores_from_hidden_states_masks_non_completion_targets() -> None:
    # Identity lm_head makes the hidden states the logits directly.
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    token_logprobs, _entropies = token_scores_from_hidden_states(
        logits,
        lm_head=torch.nn.Identity(),
        target_ids=torch.tensor([[1, 2]], dtype=torch.long),
        target_mask=torch.tensor([[False, True]]),
        compute_entropies=False,
    )

    assert torch.isclose(token_logprobs[0, 0], torch.tensor(0.0))
    expected = torch.log_softmax(logits[0, 1], dim=-1)[2]
    assert torch.isclose(token_logprobs[0, 1], expected)


def test_token_scores_from_hidden_states_entropies_mask_non_completion_targets() -> (
    None
):
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    _token_logprobs, entropies = token_scores_from_hidden_states(
        logits,
        lm_head=torch.nn.Identity(),
        target_ids=torch.tensor([[0, 0]], dtype=torch.long),
        target_mask=torch.tensor([[False, True]]),
        compute_entropies=True,
    )

    assert torch.isclose(entropies[0, 0], torch.tensor(0.0))
    assert torch.isclose(entropies[0, 1], torch.log(torch.tensor(3.0)))


def test_token_scores_from_hidden_states_projects_through_lm_head() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    lm_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.tensor([[0.5, -0.5], [1.0, 0.25], [-0.75, 2.0]]))
    target_ids = torch.tensor([[2, 0]], dtype=torch.long)
    target_mask = torch.tensor([[True, True]])

    token_logprobs, _entropies = token_scores_from_hidden_states(
        hidden,
        lm_head=lm_head,
        target_ids=target_ids,
        target_mask=target_mask,
        compute_entropies=False,
    )

    reference = torch.log_softmax(lm_head(hidden[:, :-1, :]), dim=-1)
    assert torch.isclose(token_logprobs[0, 0], reference[0, 0, 2])
    assert torch.isclose(token_logprobs[0, 1], reference[0, 1, 0])


@pytest.mark.parametrize("keep", [6, 3])
@pytest.mark.parametrize("age", [60, 3 * 3600])
def test_save_latest_adapter_uses_configured_retention(tmp_path, keep, age) -> None:
    import os
    import time as _time

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.model = _DummyAdapterModel()
    assert trainer.config.checkpoint_keep_last == 6
    trainer.config.checkpoint_keep_last = keep
    initial = trainer._save_latest_adapter(tmp_path, -1)
    assert (initial / "PINNED").exists()
    now = _time.time()
    for i in range(50):
        d = tmp_path / f"adapter_iter_{i:06d}"
        d.mkdir()
        os.utime(d, (now - age, now - age))
    os.utime(initial, (now - age, now - age))

    latest = trainer._save_latest_adapter(tmp_path, 50)

    # Fresh adapters survive up to the hard cap; old ones use keep. The
    # initial pinned snapshot is exempt from both bounds.
    retained = 40 if age < 7200 else keep
    assert sorted(tmp_path.glob("adapter_iter_*")) == [initial] + [
        tmp_path / f"adapter_iter_{i:06d}" for i in range(51 - retained, 51)
    ]
    assert (tmp_path / "latest_adapter").resolve() == latest


@pytest.mark.parametrize("age", [60, 3 * 3600])
def test_save_latest_adapter_keeps_every_adapter_when_null(
    tmp_path, monkeypatch, age
) -> None:
    import os
    import time as _time

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.model = _DummyAdapterModel()
    trainer.config.checkpoint_keep_last = None
    trainer.config.pinned_adapter_interval = 25
    monkeypatch.setattr(
        trainer, "_cleanup_old_adapters",
        lambda *_args, **_kwargs: pytest.fail("null retention must skip cleanup"),
    )
    initial = trainer._save_latest_adapter(tmp_path, -1)
    now = _time.time()
    # Exceed both the 40-pinned and 40-prunable caps, with fresh or old dirs.
    for i in range(100):
        d = tmp_path / f"adapter_iter_{i:06d}"
        d.mkdir()
        if i < 50:
            (d / "PINNED").touch()
        os.utime(d, (now - age, now - age))
    os.utime(initial, (now - age, now - age))

    latest = trainer._save_latest_adapter(tmp_path, 100)

    assert sorted(tmp_path.glob("adapter_iter_*")) == [initial] + [
        tmp_path / f"adapter_iter_{i:06d}" for i in range(101)
    ]
    assert (initial / "PINNED").exists()
    assert (latest / "PINNED").exists()
    assert (tmp_path / "latest_adapter").resolve() == latest


def test_cleanup_old_adapters_is_time_and_count_bounded(tmp_path) -> None:
    import os
    import time as _time

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    now = _time.time()
    # 10 adapters: 0-4 old (3h), 5-9 fresh. keep=2, hard_cap=8.
    for i in range(10):
        d = tmp_path / f"adapter_iter_{i:06d}"
        d.mkdir()
        age = 3 * 3600 if i < 5 else 60
        os.utime(d, (now - age, now - age))

    trainer._cleanup_old_adapters(tmp_path, keep=2, min_age_seconds=7200, hard_cap=8)
    left = sorted(p.name for p in tmp_path.glob("adapter_iter_*"))
    # 0,1 exceed the hard cap (10-8) -> deleted regardless of age.
    # 2,3,4 are beyond keep=2 and older than min_age -> deleted.
    # 5,6,7 are beyond keep=2 but younger than min_age -> KEPT (time bound).
    # 8,9 are the keep=2 newest -> kept.
    assert left == [f"adapter_iter_{i:06d}" for i in (5, 6, 7, 8, 9)]


def test_cleanup_old_adapters_never_prunes_pinned_snapshots(tmp_path) -> None:
    import os
    import time as _time

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    now = _time.time()
    for i in range(10):
        d = tmp_path / f"adapter_iter_{i:06d}"
        d.mkdir()
        if i in (0, 4, 8):
            (d / "PINNED").touch()
        age = 3 * 3600 if i < 5 else 60
        os.utime(d, (now - age, now - age))

    trainer._cleanup_old_adapters(tmp_path, keep=2, min_age_seconds=7200, hard_cap=8)
    left = sorted(p.name for p in tmp_path.glob("adapter_iter_*"))
    # Same layout as the base case, but 0, 4, 8 carry the PINNED marker: all
    # three are exempt from every rule. The bounds then act on the 7 prunable
    # dirs: 1, 2, 3 are beyond keep=2 and older than min_age -> deleted;
    # 5, 6, 7 are younger than min_age -> kept; 9 is within keep=2.
    assert left == [f"adapter_iter_{i:06d}" for i in (0, 4, 5, 6, 7, 8, 9)]


def test_cleanup_old_adapters_bounds_pinned_snapshots(tmp_path, monkeypatch) -> None:
    import os
    import time as _time

    monkeypatch.setattr("training.ppo._PINNED_ADAPTER_CAP", 2)
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    now = _time.time()
    for i in range(6):
        d = tmp_path / f"adapter_iter_{i:06d}"
        d.mkdir()
        (d / "PINNED").touch()
        os.utime(d, (now - 3 * 3600, now - 3 * 3600))

    trainer._cleanup_old_adapters(tmp_path, keep=2, min_age_seconds=7200, hard_cap=8)
    left = sorted(p.name for p in tmp_path.glob("adapter_iter_*"))
    # Only the newest 2 pinned dirs stay exempt; the older 4 fall back to the
    # normal bounds (all old, beyond keep=2 of the prunable list -> 2 kept).
    assert left == [f"adapter_iter_{i:06d}" for i in (2, 3, 4, 5)]


def test_token_scores_chunked_matches_direct_with_gradients() -> None:
    # Densely-masked sequences must produce identical scores AND input
    # gradients whether the vocab projection runs in one piece or in
    # checkpointed chunks (chunk_size=3 forces 4 chunks over 10 targets;
    # the production OOM this guards against was run 20260711_122810).
    torch.manual_seed(7)
    hidden_direct = torch.randn(1, 11, 8, requires_grad=True)
    hidden_chunked = hidden_direct.detach().clone().requires_grad_(True)
    lm_head = torch.nn.Linear(8, 13, bias=False)
    target_ids = torch.randint(0, 13, (1, 10))
    target_mask = torch.ones(1, 10, dtype=torch.bool)

    logprobs_direct, entropies_direct = token_scores_from_hidden_states(
        hidden_direct,
        lm_head=lm_head,
        target_ids=target_ids,
        target_mask=target_mask,
        compute_entropies=True,
        chunk_size=10_000,
    )
    logprobs_chunked, entropies_chunked = token_scores_from_hidden_states(
        hidden_chunked,
        lm_head=lm_head,
        target_ids=target_ids,
        target_mask=target_mask,
        compute_entropies=True,
        chunk_size=3,
    )

    assert torch.allclose(logprobs_chunked, logprobs_direct, atol=1e-6)
    assert torch.allclose(entropies_chunked, entropies_direct, atol=1e-6)

    (logprobs_direct.sum() + entropies_direct.sum()).backward()
    (logprobs_chunked.sum() + entropies_chunked.sum()).backward()
    assert hidden_direct.grad is not None and hidden_chunked.grad is not None
    assert torch.allclose(hidden_chunked.grad, hidden_direct.grad, atol=1e-6)


def test_action_efficiency_counts_only_completed_levels() -> None:
    solved = RolloutTrajectory(
        trajectory_id="solved",
        task_id="ar25",
        steps=[],
        metadata={
            "start_level": 0,
            "levels_completed": 2,
            "run_total_actions": 20,
        },
    )
    unsolved = RolloutTrajectory(
        trajectory_id="unsolved",
        task_id="ar25",
        steps=[],
        metadata={
            "start_level": 0,
            "levels_completed": 0,
            "run_total_actions": 10,
        },
    )

    # Both episodes consume actions; only the first earns two new level solves.
    assert _avg_actions_per_completed_level([solved, unsolved]) == 15.0
    assert _avg_actions_per_completed_level([unsolved]) == 0.0


def test_summarize_sample_lengths_reports_input_and_completion_percentiles() -> None:
    samples = [
        TokenizedSample(
            trajectory_id="t0",
            step_index=0,
            input_ids=[1, 2, 3, 4],
            labels=[-100, 2, 3, 4],
            attention_mask=[1, 1, 1, 1],
            completion_mask=[0, 1, 1, 1],
            reward=1.0,
            return_to_go=1.0,
            advantage=0.5,
        ),
        TokenizedSample(
            trajectory_id="t1",
            step_index=1,
            input_ids=[1, 2, 3, 4, 5, 6],
            labels=[-100, -100, 3, 4, 5, 6],
            attention_mask=[1, 1, 1, 1, 1, 1],
            completion_mask=[0, 0, 1, 1, 1, 1],
            reward=0.5,
            return_to_go=0.5,
            advantage=0.1,
        ),
        TokenizedSample(
            trajectory_id="t2",
            step_index=2,
            input_ids=[1, 2, 3, 4, 5],
            labels=[-100, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            completion_mask=[0, 1, 1, 1, 1],
            reward=0.2,
            return_to_go=0.2,
            advantage=-0.1,
        ),
    ]

    summary = summarize_sample_lengths(samples)

    assert summary.sample_count == 3
    assert summary.min_input_tokens == 4
    assert summary.median_input_tokens == 5
    assert summary.p95_input_tokens == 6
    assert summary.max_input_tokens == 6
    assert summary.min_completion_tokens == 3
    assert summary.median_completion_tokens == 4
    assert summary.p95_completion_tokens == 4
    assert summary.max_completion_tokens == 4


def test_filter_zero_signal_samples_skips_zero_advantage_samples() -> None:
    samples = [
        TokenizedSample(
            trajectory_id="t0",
            step_index=0,
            input_ids=[1, 2],
            labels=[-100, 2],
            attention_mask=[1, 1],
            completion_mask=[0, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=0.0,
        ),
        TokenizedSample(
            trajectory_id="t1",
            step_index=1,
            input_ids=[1, 3],
            labels=[-100, 3],
            attention_mask=[1, 1],
            completion_mask=[0, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=-0.1,
        ),
        TokenizedSample(
            trajectory_id="t2",
            step_index=2,
            input_ids=[1, 4],
            labels=[-100, 4],
            attention_mask=[1, 1],
            completion_mask=[0, 1],
            reward=0.25,
            return_to_go=0.25,
            advantage=0.0,
        ),
    ]

    retained, skipped = _filter_zero_signal_samples(
        samples,
        previous_policy_kl_coef=0.0,
        reference_kl_coef=0.0,
        entropy_coef=0.0,
    )

    assert skipped == 2
    assert retained == [samples[1]]


@pytest.mark.parametrize(
    "auxiliary_loss", ["previous_policy_kl", "reference_kl", "entropy", "disabled"]
)
def test_zero_signal_filter_preserves_full_loss_gradient_and_update(
    auxiliary_loss: str,
) -> None:
    from training.ppo import _zero_signal_denominator_scale

    trainer = _make_tiny_ppo_trainer(advantage=0.0)
    trainer.config.previous_policy_kl_coef = (
        0.2 if auxiliary_loss == "previous_policy_kl" else 0.0
    )
    trainer.config.reference_kl_coef = 0.3 if auxiliary_loss == "reference_kl" else 0.0
    # Exercise the live coefficient even though config.entropy_coef is zero.
    trainer.live_entropy_coef = 0.1 if auxiliary_loss == "entropy" else 0.0
    trainer._reference_token_logprobs = lambda batch: torch.full(
        (batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1),
        -math.log(5.0),
    )
    samples = [
        _tiny_action_sample(advantage=0.4, action_token_id=2),
        _tiny_action_sample(advantage=0.0, action_token_id=3),
    ]
    samples[0].sequence_weight = 0.25
    samples[1].sequence_weight = 0.75
    # Stay inside the PPO clip interval so the disabled-auxiliary case also
    # compares a real nonzero policy gradient and parameter update.
    initial_logits = torch.tensor([0.05, -0.04, 0.1, -0.08, 0.02])
    raw_mass = sum(sample.sequence_weight for sample in samples)

    def gradient_and_update(chunk, denominator):
        with torch.no_grad():
            trainer.model.logits.copy_(initial_logits)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        trainer._micro_batch_forward_backward(chunk, loss_denominator=denominator)
        gradient = trainer.model.logits.grad.detach().clone()
        optimizer.step()
        return gradient, trainer.model.logits.detach().clone()

    full_gradient, full_update = gradient_and_update(samples, raw_mass)
    assert torch.linalg.vector_norm(full_gradient).item() > 1e-4
    assert torch.linalg.vector_norm(full_update - initial_logits).item() > 1e-5
    retained, skipped = _filter_zero_signal_samples(
        samples,
        previous_policy_kl_coef=trainer.config.previous_policy_kl_coef,
        reference_kl_coef=trainer.config.reference_kl_coef,
        entropy_coef=trainer.live_entropy_coef,
    )
    retained_mass = sum(sample.sequence_weight for sample in retained)
    denominator = retained_mass * _zero_signal_denominator_scale(
        retained_count=retained_mass, filtered_count=raw_mass - retained_mass
    )
    filtered_gradient, filtered_update = gradient_and_update(retained, denominator)

    torch.testing.assert_close(filtered_gradient, full_gradient)
    torch.testing.assert_close(filtered_update, full_update)
    # The old advantage-only filter genuinely changes the auxiliary objective.
    old_gradient, old_update = gradient_and_update(samples[:1], raw_mass)
    if auxiliary_loss == "disabled":
        assert skipped == 1
        torch.testing.assert_close(old_gradient, full_gradient)
        torch.testing.assert_close(old_update, full_update)
    else:
        assert skipped == 0
        assert torch.linalg.vector_norm(old_gradient - full_gradient).item() > 1e-4
        assert torch.linalg.vector_norm(old_update - full_update).item() > 1e-5


@pytest.mark.parametrize(
    "auxiliary_loss", ["previous_policy_kl", "reference_kl", "entropy", "disabled"]
)
def test_all_zero_advantage_window_trains_only_active_auxiliary_loss(
    monkeypatch, tmp_path: Path, auxiliary_loss: str
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=0.0, learning_rate=0.01)
    trainer.config.previous_policy_kl_coef = (
        0.2 if auxiliary_loss == "previous_policy_kl" else 0.0
    )
    trainer.config.reference_kl_coef = 0.3 if auxiliary_loss == "reference_kl" else 0.0
    trainer._reference_token_logprobs = lambda batch: torch.full(
        (batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1),
        -math.log(5.0),
    )
    # The rollout came from the uniform policy; one intervening update moved
    # the actor. Keep the real same-checkpoint parity path for this stale data.
    trainer._remember_parity_policy(-1)
    with torch.no_grad():
        trainer.model.logits[2] = 0.1
    before = trainer.model.logits.detach().clone()
    sample = _tiny_action_sample(advantage=0.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000001.pt",
    )

    result = trainer.train_window(
        [_single_step_rollout()],
        output_dir=tmp_path,
        iteration_index=1,
        policy_lag_windows=1,
        entropy_coef=0.1 if auxiliary_loss == "entropy" else 0.0,
    )

    assert result.raw_sample_count == 1
    assert result.zero_signal_denominator_scale == 1.0
    if auxiliary_loss == "disabled":
        assert result.filtered_zero_signal_sample_count == 1
        assert result.optimizer_steps == 0
        assert result.processed_tokens == 0
        assert result.forward_time_s == result.backward_time_s == 0.0
        assert result.optimizer_cycle_seconds == 0.0
        assert result.optimization_seconds >= 0.0
        assert result.optimizer_skipped_reason == "zero_nonzero_advantage_samples"
        torch.testing.assert_close(trainer.model.logits, before, rtol=0.0, atol=0.0)
    else:
        assert result.filtered_zero_signal_sample_count == 0
        assert result.eligible_sample_count == result.sample_count == 1
        assert result.optimizer_steps == 1
        assert result.optimizer_skipped_reason == ""
        assert result.mean_grad_norm > 0.0
        # All three auxiliary terms pull this overconfident action back toward
        # the uniform old/reference policy (or maximum entropy).
        assert trainer.model.logits[2].item() < before[2].item()


def test_zero_signal_denominator_scale_math() -> None:
    from training.ppo import _zero_signal_denominator_scale

    # (retained + filtered) / retained restores the pre-filter average.
    assert _zero_signal_denominator_scale(
        retained_count=6, filtered_count=2
    ) == pytest.approx(8.0 / 6.0)
    assert _zero_signal_denominator_scale(
        retained_count=1, filtered_count=3
    ) == pytest.approx(4.0)
    # Nothing filtered -> exactly 1.0 (no behavior change).
    assert _zero_signal_denominator_scale(retained_count=4, filtered_count=0) == 1.0
    # Degenerate windows guard to 1.0 instead of dividing by zero.
    assert _zero_signal_denominator_scale(retained_count=0, filtered_count=5) == 1.0
    assert _zero_signal_denominator_scale(retained_count=0, filtered_count=0) == 1.0


@pytest.mark.parametrize("episode_packing, filtered_segments", [(False, 1), (True, 3)])
def test_train_window_scales_loss_denominator_for_filtered_samples(
    monkeypatch, tmp_path: Path, episode_packing: bool, filtered_segments: int
) -> None:
    # One retained + one zero-signal sample: every cycle's loss denominator
    # must be multiplied by (1 + 1) / 1 = 2 so the aggregate update is as if
    # the zero-gradient span were still present in the average.
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.episode_packing = episode_packing
    retained = _tiny_action_sample(advantage=1.0)
    filtered = [_tiny_action_sample(advantage=0.0) for _ in range(filtered_segments)]
    for index, sample in enumerate(filtered):
        sample.trajectory_id = "filtered"
        sample.step_index = index
        sample.segment_start = index > 0
    monkeypatch.setattr(
        "training.ppo.build_samples", lambda *args, **kwargs: [retained, *filtered]
    )
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )
    seen_denominators: list[float] = []
    original_forward_backward = trainer._micro_batch_forward_backward

    def recording_forward_backward(chunk, *, loss_denominator, **kwargs):
        seen_denominators.append(loss_denominator)
        return original_forward_backward(
            chunk, loss_denominator=loss_denominator, **kwargs
        )

    monkeypatch.setattr(
        trainer, "_micro_batch_forward_backward", recording_forward_backward
    )

    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )

    assert result.filtered_zero_signal_sample_count == filtered_segments
    assert result.zero_signal_denominator_scale == pytest.approx(1 + filtered_segments)
    assert seen_denominators == [pytest.approx(1 + filtered_segments)]


def test_sample_budget_count_rounds_down_to_optimizer_step_multiple() -> None:
    assert _sample_budget_count(117, fraction=0.8, step_multiple=32) == 64
    assert _sample_budget_count(117, fraction=1.0, step_multiple=32) == 117
    assert _sample_budget_count(20, fraction=0.5, step_multiple=32) == 10
    assert _sample_budget_count(117, fraction=0.01, step_multiple=32) == 32


def test_sample_budget_count_honors_zero_fraction() -> None:
    assert _sample_budget_count(117, fraction=0.0, step_multiple=32) == 0
    assert _sample_budget_count(117, fraction=-0.1, step_multiple=32) == 0


def test_extra_cycle_guard_requires_rollout_eta_and_safety_margin() -> None:
    assert (
        _extra_cycle_block_reason(
            next_window_ready=lambda: False,
            next_window_seconds_remaining=None,
            estimated_cycle_seconds=100.0,
        )
        == "rollout_eta_unavailable"
    )
    assert (
        _extra_cycle_block_reason(
            next_window_ready=lambda: False,
            next_window_seconds_remaining=lambda: 119.9,
            estimated_cycle_seconds=100.0,
        )
        == "insufficient_rollout_slack"
    )
    assert (
        _extra_cycle_block_reason(
            next_window_ready=lambda: False,
            next_window_seconds_remaining=lambda: 120.0,
            estimated_cycle_seconds=100.0,
        )
        == ""
    )
    assert (
        _extra_cycle_block_reason(
            next_window_ready=lambda: True,
            next_window_seconds_remaining=lambda: 1000.0,
            estimated_cycle_seconds=100.0,
        )
        == "window_ready"
    )


def test_subsample_selection_is_independent_of_episode_reward() -> None:
    import random

    samples = [_sample_with_input_length(f"t{i}", 2) for i in range(10)]
    for seed in range(20):
        random.seed(seed)
        retained, skipped = _subsample_train_samples(
            samples,
            fraction=0.5,
            step_multiple=1,
        )
        assert skipped == 5
        kept_ids = {sample.trajectory_id for sample in retained}
        for index, sample in enumerate(samples):
            sample.reward = 20.0 if index in (3, 7) else 0.0
            sample.sequence_weight = 1.0
        random.seed(seed)
        rerun, _ = _subsample_train_samples(samples, fraction=0.5, step_multiple=1)
        assert {sample.trajectory_id for sample in rerun} == kept_ids
        for sample in samples:
            sample.reward = 0.0
            sample.sequence_weight = 1.0


def test_subsample_stratifies_per_episode_with_floor_and_inclusion_weights() -> None:
    # Stratifies by EPISODE: every episode keeps >= 1 segment (floor),
    # quotas are proportional to eligible counts (largest remainder), and the
    # survivors carry Horvitz-Thompson inclusion weights (eligible/kept) so
    # the subsample does not distort within-episode composition in
    # expectation. A global |advantage| ranking would drop episode B
    # entirely; stratification cannot.
    samples = []
    for episode_id, count, advantage in (("ep-A", 9, 5.0), ("ep-B", 3, 0.01)):
        for i in range(count):
            sample = _sample_with_input_length(f"{episode_id}", 2)
            sample.trajectory_id = episode_id
            sample.task_id = f"game-{episode_id}"
            sample.reward = 0.0
            sample.advantage = advantage
            samples.append(sample)

    for _attempt in range(25):
        for sample in samples:
            sample.sequence_weight = 1.0
        retained, skipped = _subsample_train_samples(
            samples,
            fraction=0.5,
            step_multiple=1,
        )
        assert skipped == 6
        by_episode = {"ep-A": 0, "ep-B": 0}
        for sample in retained:
            by_episode[sample.trajectory_id] += 1
        # Deterministic quotas: floor 1 each, then largest-remainder over
        # capacities {A: 8, B: 2} for 4 spare -> A=4, B=2.
        assert by_episode == {"ep-A": 4, "ep-B": 2}
        for sample in retained:
            expected = 9 / 4 if sample.trajectory_id == "ep-A" else 3 / 2
            assert sample.sequence_weight == pytest.approx(expected)


@pytest.mark.parametrize("same_episode", [False, True])
def test_subsample_preserves_expected_episode_objective(
    monkeypatch, same_episode: bool
) -> None:
    from training.ppo import _window_loss_denominator

    full = [
        _weighted_loss_sample(
            "first", advantage=3.0, weight=0.1 if same_episode else 1.0
        ),
        _weighted_loss_sample(
            "first" if same_episode else "second",
            advantage=-1.5,
            weight=0.9 if same_episode else 1.0,
        ),
    ]
    expected_mass = sum(s.sequence_weight for s in full)
    expected = sum(s.advantage * s.sequence_weight for s in full) / expected_mass
    estimates = []
    from dataclasses import replace

    # Enumerate the two equiprobable draws, avoiding a statistical tolerance.
    for pick_last in (False, True):
        monkeypatch.setattr(
            "training.ppo.random.sample",
            lambda population, k: list(
                population[-k:] if pick_last and k else population[:k]
            ),
        )
        retained, _ = _subsample_train_samples(
            [replace(s) for s in full],
            fraction=0.5,
            step_multiple=1,
        )
        denominator = _window_loss_denominator(
            retained,
            micro_batch_count=1,
            cycle_size=1,
            expected_mass=expected_mass,
        )
        estimates.append(
            sum(s.advantage * s.sequence_weight for s in retained) / denominator
        )
    assert sum(estimates) / len(estimates) == pytest.approx(expected)


def _sample_with_input_length(trajectory_id: str, length: int) -> TokenizedSample:
    input_ids = list(range(1, length + 1))
    return TokenizedSample(
        trajectory_id=trajectory_id,
        step_index=0,
        input_ids=input_ids,
        labels=[-100, *input_ids[1:]],
        attention_mask=[1] * length,
        completion_mask=[0, *([1] * max(0, length - 1))],
        reward=0.0,
        return_to_go=0.0,
        advantage=0.0,
    )


def test_batch_iter_respects_max_micro_batch_tokens(monkeypatch) -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="tiny",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=16,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        max_micro_batch_tokens=10,
    )
    samples = [
        _sample_with_input_length("short-0", 3),
        _sample_with_input_length("short-1", 4),
        _sample_with_input_length("long-0", 6),
        _sample_with_input_length("too-large-for-pair", 12),
    ]
    monkeypatch.setattr("training.ppo.random.shuffle", lambda _items: None)

    batches = [
        [sample.trajectory_id for sample in batch]
        for batch in trainer._batch_iter(samples)
    ]

    assert batches == [
        ["too-large-for-pair"],
        ["long-0"],
        ["short-1", "short-0"],
    ]


def test_train_window_warns_and_skips_optimizer_when_no_samples(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        value_baseline="hidden_critic",
        model_name="Qwen/Qwen3.5-9B",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=512,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=0.0,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=8, alpha=16, dropout=0.0, target_modules=["q_proj"]),
    )
    trainer.model = _TinyPolicy()
    trainer.device = torch.device("cpu")
    trainer.tokenizer = object()
    trainer.optimizer = type(
        "_DummyOptimizer",
        (),
        {
            "param_groups": [{"lr": 0.0}],
            "state_dict": lambda _self: {"state": {}, "param_groups": []},
        },
    )()
    trainer._optimizer_step_count = 0
    trainer._trainable_parameter_count = 123
    trainer.state_critic = _test_state_critic()
    trainer._extract_policy_hidden_states = lambda _rollouts: {}

    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [])
    monkeypatch.setattr(trainer, "_trainable_parameter_l2_norm", lambda: 1.25)
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter_iter_000000",
            output_dir / "adapter_iter_000000" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    with caplog.at_level(logging.WARNING, logger="training.ppo"):
        result = trainer.train_window(
            [
                RolloutTrajectory(
                    trajectory_id="empty",
                    task_id="pathfinder-0001",
                    steps=[],
                    metadata={"run_total_actions": 0},
                )
            ],
            output_dir=tmp_path,
            iteration_index=0,
        )

    assert "zero nonzero-advantage samples" in caplog.text
    assert result.sample_count == 0
    assert result.raw_sample_count == 0
    assert result.optimizer_steps == 0
    assert result.optimizer_skipped_reason == "zero_nonzero_advantage_samples"
    assert result.adapter_dir == tmp_path / "adapter_iter_000000"
    assert (
        result.runtime_adapter_dir == tmp_path / "adapter_iter_000000" / "runtime_vllm"
    )


def test_training_state_round_trips_optimizer_step(monkeypatch, tmp_path: Path) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    trainer._optimizer_step_count = 7
    adapter_dir = tmp_path / "adapter_iter_000003"
    runtime_adapter_dir = adapter_dir / "runtime_vllm"
    runtime_adapter_dir.mkdir(parents=True)

    state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=3,
        adapter_dir=adapter_dir,
        runtime_adapter_dir=runtime_adapter_dir,
    )

    restored = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    restored.load_training_state(state_path)

    assert state_path == tmp_path / "trainer_state_iter_000003.pt"
    assert (tmp_path / "latest_trainer_state.pt").exists()
    assert restored.optimizer_step_count == 7


def test_training_state_save_preserves_previous_checkpoint_for_resume(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    trainer.config.checkpoint_keep_last = 2
    first_adapter_dir = tmp_path / "adapter_iter_000003"
    first_runtime_adapter_dir = first_adapter_dir / "runtime_vllm"
    first_runtime_adapter_dir.mkdir(parents=True)
    second_adapter_dir = tmp_path / "adapter_iter_000004"
    second_runtime_adapter_dir = second_adapter_dir / "runtime_vllm"
    second_runtime_adapter_dir.mkdir(parents=True)
    third_adapter_dir = tmp_path / "adapter_iter_000005"
    third_runtime_adapter_dir = third_adapter_dir / "runtime_vllm"
    third_runtime_adapter_dir.mkdir(parents=True)

    first_state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=3,
        adapter_dir=first_adapter_dir,
        runtime_adapter_dir=first_runtime_adapter_dir,
    )
    trainer._optimizer_step_count = 11
    second_state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=4,
        adapter_dir=second_adapter_dir,
        runtime_adapter_dir=second_runtime_adapter_dir,
    )

    restored = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    restored.load_training_state(second_state_path)

    assert first_state_path.exists()
    assert sorted(tmp_path.glob("trainer_state_iter_*.pt")) == [
        first_state_path,
        second_state_path,
    ]
    assert (tmp_path / "latest_trainer_state.pt").exists()
    assert restored.optimizer_step_count == 11

    trainer._optimizer_step_count = 13
    third_state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=5,
        adapter_dir=third_adapter_dir,
        runtime_adapter_dir=third_runtime_adapter_dir,
    )

    assert not first_state_path.exists()
    assert sorted(tmp_path.glob("trainer_state_iter_*.pt")) == [
        second_state_path,
        third_state_path,
    ]


@pytest.mark.parametrize("save_initial_state", [False, True])
@pytest.mark.parametrize("keep", [6, 2, None])
def test_training_state_save_uses_configured_retention(
    monkeypatch, tmp_path: Path, save_initial_state: bool, keep: int | None
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    assert trainer.config.checkpoint_keep_last == 6
    trainer.config.checkpoint_keep_last = keep
    if keep is None:
        monkeypatch.setattr(
            trainer, "_cleanup_old_training_states",
            lambda *_args, **_kwargs: pytest.fail("null retention must skip cleanup"),
        )
    saved_states: dict[int, Path] = {}
    initial_bytes = None
    for iteration in ([-1] if save_initial_state else []) + list(range(9)):
        adapter_dir = tmp_path / f"adapter_iter_{iteration:06d}"
        runtime_adapter_dir = adapter_dir / "runtime_vllm"
        runtime_adapter_dir.mkdir(parents=True)
        trainer._optimizer_step_count = iteration + 1
        saved_states[iteration] = trainer._save_training_state(
            tmp_path,
            iteration_index=iteration,
            adapter_dir=adapter_dir,
            runtime_adapter_dir=runtime_adapter_dir,
        )
        if iteration == -1:
            initial_bytes = saved_states[-1].read_bytes()
        first_retained = 0 if keep is None else max(0, iteration + 1 - keep)
        expected_iterations = set(range(first_retained, iteration + 1))
        if save_initial_state:
            expected_iterations.add(-1)
        assert set(tmp_path.glob("trainer_state_iter_*.pt")) == {
            saved_states[index] for index in expected_iterations
        }
        assert (tmp_path / "latest_trainer_state.pt").resolve() == saved_states[iteration]

    if save_initial_state:
        assert saved_states[-1].read_bytes() == initial_bytes
    restored = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    restored.load_training_state(tmp_path / "latest_trainer_state.pt")
    assert restored.optimizer_step_count == 9


def test_training_state_cleanup_sorts_by_iteration_number(tmp_path: Path) -> None:
    older_state_path = tmp_path / "trainer_state_iter_999999.pt"
    latest_state_path = tmp_path / "trainer_state_iter_1000000.pt"
    older_state_path.write_text("older")
    latest_state_path.write_text("latest")

    PPOTrainer._cleanup_old_training_states(tmp_path, keep=1)

    assert not older_state_path.exists()
    assert latest_state_path.exists()


def test_resume_uses_optimizer_step_for_warmup(monkeypatch, tmp_path: Path) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    trainer.config.warmup_steps = 10
    trainer._optimizer_step_count = 7
    adapter_dir = tmp_path / "adapter_iter_000003"
    runtime_adapter_dir = adapter_dir / "runtime_vllm"
    runtime_adapter_dir.mkdir(parents=True)

    state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=3,
        adapter_dir=adapter_dir,
        runtime_adapter_dir=runtime_adapter_dir,
    )

    restored = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    restored.config.warmup_steps = 10
    restored.load_training_state(state_path)
    resumed_learning_rate = restored._set_optimizer_learning_rate(
        restored.optimizer_step_count + 1
    )

    assert restored.optimizer_step_count == 7
    assert resumed_learning_rate == pytest.approx(0.04)


def test_training_state_does_not_persist_state_critic_replay(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    adapter_dir = tmp_path / "adapter_iter_000005"
    runtime_adapter_dir = adapter_dir / "runtime_vllm"
    runtime_adapter_dir.mkdir(parents=True)

    state_path = trainer._save_training_state(
        tmp_path,
        iteration_index=5,
        adapter_dir=adapter_dir,
        runtime_adapter_dir=runtime_adapter_dir,
    )

    state = torch.load(state_path, map_location="cpu", weights_only=False)

    assert "state_critic_replay" not in state


def test_exported_runtime_adapter_stays_identical_when_learning_rate_is_zero(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.0)

    initial_adapter_dir, initial_runtime_dir, _ = trainer._export_policy_snapshot(
        tmp_path, 0
    )
    initial_adapter_tensors = _load_adapter_tensors(initial_adapter_dir)
    initial_runtime_tensors = _load_adapter_tensors(initial_runtime_dir)

    result = trainer.train_window(
        [_dummy_rollout()],
        output_dir=tmp_path,
        iteration_index=1,
    )

    assert result.optimizer_steps == 1
    assert result.trainable_param_norm_before == 0.0
    assert result.trainable_param_norm_after == 0.0
    assert _tensor_dicts_match(
        initial_adapter_tensors, _load_adapter_tensors(result.adapter_dir)
    )
    assert _tensor_dicts_match(
        initial_runtime_tensors, _load_adapter_tensors(result.runtime_adapter_dir)
    )
    assert runtime_lora_has_effective_delta(result.runtime_adapter_dir) is False


def test_train_window_rejects_overlong_reward_sample(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.0)
    long_positive_reward_sample = TokenizedSample(
        trajectory_id="long-positive",
        step_index=3,
        input_ids=list(range(20)),
        labels=[-100] * 19 + [1],
        attention_mask=[1] * 20,
        completion_mask=[0] * 19 + [1],
        reward=2.0,
        return_to_go=2.0,
        advantage=1.0,
    )
    kept_credit_sample = TokenizedSample(
        trajectory_id="kept-credit",
        step_index=0,
        input_ids=[1, 2],
        labels=[-100, 2],
        attention_mask=[1, 1],
        completion_mask=[0, 1],
        reward=0.0,
        return_to_go=2.0,
        advantage=1.0,
    )
    monkeypatch.setattr(
        "training.ppo.build_samples",
        lambda *args, **kwargs: [long_positive_reward_sample, kept_credit_sample],
    )

    with pytest.raises(ValueError, match="max_sequence_tokens=16"):
        trainer.train_window(
            [_dummy_rollout()],
            output_dir=tmp_path,
            iteration_index=1,
        )


def test_overlong_samples_fail_before_baseline_update(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.0)
    trainer.config.max_sequence_tokens = 4
    before_baseline = {
        k: v.clone() for k, v in trainer.state_critic.network.state_dict().items()
    }
    overlong_sample = TokenizedSample(
        trajectory_id="long-policy-sample",
        step_index=0,
        input_ids=list(range(20)),
        labels=[-100] * 19 + [1],
        attention_mask=[1] * 20,
        completion_mask=[0] * 19 + [1],
        reward=10.0,
        return_to_go=30.0,
        advantage=1.0,
    )
    monkeypatch.setattr(
        "training.ppo.build_samples",
        lambda *args, **kwargs: [overlong_sample],
    )
    rollout = RolloutTrajectory(
        trajectory_id="critic-rollout",
        task_id="pathfinder-0001",
        trajectory_reward=30.0,
        metadata={"game_total_levels": 3, "max_runtime_seconds": 300.0},
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=10.0,
                done=False,
                metadata={
                    "stage": {
                        "levels_completed_before_decision": 0,
                        "levels_remaining": 3,
                        "time_remaining_seconds": 300.0,
                    }
                },
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=20.0,
                done=True,
                metadata={
                    "stage": {
                        "levels_completed_before_decision": 1,
                        "levels_remaining": 2,
                        "time_remaining_seconds": 240.0,
                    }
                },
            ),
        ],
    )

    with pytest.raises(ValueError, match="max_sequence_tokens=4"):
        trainer.train_window([rollout], output_dir=tmp_path, iteration_index=1)
    assert all(
        torch.equal(v, before_baseline[k])
        for k, v in trainer.state_critic.network.state_dict().items()
    )


def test_exported_runtime_adapter_changes_when_learning_rate_is_positive(
    monkeypatch, tmp_path: Path
) -> None:
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)

    initial_adapter_dir, initial_runtime_dir, _ = trainer._export_policy_snapshot(
        tmp_path, 0
    )
    initial_adapter_tensors = _load_adapter_tensors(initial_adapter_dir)
    initial_runtime_tensors = _load_adapter_tensors(initial_runtime_dir)

    result = trainer.train_window(
        [_dummy_rollout()],
        output_dir=tmp_path,
        iteration_index=1,
    )

    assert result.optimizer_steps == 1
    assert result.trainable_param_norm_after > result.trainable_param_norm_before
    assert not _tensor_dicts_match(
        initial_adapter_tensors, _load_adapter_tensors(result.adapter_dir)
    )
    assert not _tensor_dicts_match(
        initial_runtime_tensors, _load_adapter_tensors(result.runtime_adapter_dir)
    )
    assert runtime_lora_has_effective_delta(result.runtime_adapter_dir) is True


def _tiny_loss_test_trainer() -> PPOTrainer:
    from types import SimpleNamespace

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="test-model",
        output_dir="runs",
        device="cpu",
        seed=7,
        max_sequence_tokens=64,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._optimizer_step_count = 0
    return trainer


def _weighted_loss_sample(
    trajectory_id: str, advantage: float, weight: float
) -> TokenizedSample:
    # prompt [1,2], completion [3,4] -> 4 input tokens, 3 causal targets.
    return TokenizedSample(
        trajectory_id=trajectory_id,
        step_index=0,
        input_ids=[1, 2, 3, 4],
        labels=[-100, -100, 3, 4],
        attention_mask=[1, 1, 1, 1],
        completion_mask=[0, 0, 1, 1],
        reward=0.0,
        return_to_go=0.0,
        advantage=advantage,
        prompt_token_count=2,
        old_token_logprobs=[0.0, 0.0, 0.0],
        sequence_weight=weight,
    )


def _run_loss_only_micro_batch(
    trainer: PPOTrainer,
    chunk,
    loss_denominator=1.0,
    *,
    new_logprob_value: float = 0.0,
):
    grads: list[torch.Tensor] = []

    def fake_token_logprobs(batch, **_kwargs):
        target_len = batch["input_ids"].shape[1] - 1
        out = torch.full(
            (batch["input_ids"].shape[0], target_len),
            new_logprob_value,
            requires_grad=True,
        )
        grads.append(out)
        return out

    def fake_token_logprobs_and_entropies(batch, **_kwargs):
        logprobs = fake_token_logprobs(batch)
        return logprobs, torch.zeros_like(logprobs)

    trainer._token_logprobs_and_entropies = fake_token_logprobs_and_entropies
    return trainer._micro_batch_forward_backward(
        chunk, loss_denominator=loss_denominator
    )


def test_weighted_loss_uses_episode_equal_sequence_weights() -> None:
    trainer = _tiny_loss_test_trainer()
    # Two segments of one episode: weights 1/3 and 2/3, advantages 3.0 / -1.5.
    chunk = [
        _weighted_loss_sample("ep", advantage=3.0, weight=1.0 / 3.0),
        _weighted_loss_sample("ep", advantage=-1.5, weight=2.0 / 3.0),
    ]
    metrics = _run_loss_only_micro_batch(trainer, chunk)
    # ratio == 1 everywhere so span policy loss == -advantage; weighted sum:
    expected = -(1.0 / 3.0 * 3.0 + 2.0 / 3.0 * -1.5)
    assert abs(metrics["loss"] - expected) < 1e-6


def test_unweighted_loss_matches_weighted_sum_semantics() -> None:
    # Unit weights run the same code path as explicit weights: each sequence
    # contributes its span-mean times its (unit) mass, and the micro-batch
    # loss is the SUM over sequences — the magnitude the episode-equal
    # window denominator assumes. The legacy fallback took a global span
    # mean instead, silently rescaling by batch size.
    trainer = _tiny_loss_test_trainer()
    chunk = [
        _weighted_loss_sample("a", advantage=3.0, weight=1.0),
        _weighted_loss_sample("b", advantage=-1.5, weight=1.0),
    ]
    metrics = _run_loss_only_micro_batch(trainer, chunk)
    expected = -3.0 + 1.5
    assert abs(metrics["loss"] - expected) < 1e-6


def test_ppo_fails_on_one_nonfinite_rollout_logprob() -> None:
    trainer = _tiny_loss_test_trainer()
    sample = _weighted_loss_sample("bad-old", advantage=1.0, weight=1.0)
    sample.old_token_logprobs[-1] = float("nan")

    with pytest.raises(RuntimeError, match="rollout logprobs contain 1 non-finite"):
        _run_loss_only_micro_batch(trainer, [sample])


def test_ppo_fails_on_one_nonfinite_current_policy_logprob() -> None:
    trainer = _tiny_loss_test_trainer()
    sample = _weighted_loss_sample("bad-new", advantage=1.0, weight=1.0)

    with pytest.raises(
        RuntimeError, match="current-policy logprobs contain 2 non-finite"
    ):
        _run_loss_only_micro_batch(
            trainer,
            [sample],
            new_logprob_value=float("nan"),
        )


def test_ppo_keeps_extreme_but_finite_ratios_for_clipped_objective() -> None:
    trainer = _tiny_loss_test_trainer()
    sample = _weighted_loss_sample("finite-drift", advantage=1.0, weight=1.0)

    metrics = _run_loss_only_micro_batch(
        trainer,
        [sample],
        new_logprob_value=-10.0,
    )

    assert metrics["clip_fraction"] == pytest.approx(1.0)
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_execute_train_cycle_steps_optimizer_and_reports_clip_fraction() -> None:
    from training.memory_profile import NullMemoryLogger

    trainer = _tiny_loss_test_trainer()
    trainer._ddp_group = None
    trainer._optimizer_step_count = 0
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.zeros(2)
    trainer._trainable_parameters = lambda: [param]
    trainer._set_optimizer_learning_rate = lambda step: 1e-5

    step_calls = {"step": 0, "zero": 0}

    class _FakeOptimizer:
        def step(self):
            step_calls["step"] += 1

        def zero_grad(self, set_to_none=True):
            del set_to_none
            step_calls["zero"] += 1

    trainer.optimizer = _FakeOptimizer()
    seen_denominators: list[float] = []

    def fake_micro_batch(chunk, *, loss_denominator, **kwargs):
        seen_denominators.append(loss_denominator)
        return {
            "mean_ratio": 1.0,
            "clip_fraction": 0.25,
            "entropy": 0.0,
            "entropy_loss": 0.0,
            "loss": 0.5,
            "spans": 1.0,
        }

    trainer._micro_batch_forward_backward = fake_micro_batch
    accumulators = {
        "mean_ratio_sum": 0.0,
        "clip_fraction_sum": 0.0,
        "entropy_sum": 0.0,
        "entropy_loss_sum": 0.0,
        "ratio_updates": 0,
        "grad_norm_sum": 0.0,
        "max_grad_norm": 0.0,
        "grad_norm_updates": 0,
        "final_loss": 0.0,
        "global_step": 0,
    }
    sample = _weighted_loss_sample("ep", advantage=1.0, weight=0.5)

    clip_fraction = trainer._execute_train_cycle(
        [[sample], [sample]],
        cycle_index=1,
        epoch_label="extra pass",
        epoch_number=2,
        iteration_index=0,
        loss_denominator=1.0,
        world_size=1,
        mem=NullMemoryLogger(),
        accumulators=accumulators,
        expected_optimizer_steps_total=1,
    )

    assert clip_fraction == 0.25
    assert step_calls["step"] == 1
    assert accumulators["global_step"] == 1
    assert accumulators["ratio_updates"] == 2
    assert trainer._optimizer_step_count == 1
    assert seen_denominators == [1.0, 1.0]


def test_window_loss_denominator_is_fixed_across_tail_cycles() -> None:
    from types import SimpleNamespace
    from training.ppo import _window_loss_denominator

    # 48 full-weight episodes + one episode split into two 0.5-weight
    # compaction segments -> 50 sequences over 2 cycles of size 48 (the
    # production shape the review flagged). Per-cycle normalization would give
    # the 2-sequence tail (mass 1.0... or a lone 0.5 segment) a full-scale
    # optimizer step; the fixed denominator is the same for both cycles.
    samples = [SimpleNamespace(sequence_weight=1.0) for _ in range(48)]
    samples += [
        SimpleNamespace(sequence_weight=0.5),
        SimpleNamespace(sequence_weight=0.5),
    ]
    denominator = _window_loss_denominator(samples, micro_batch_count=50, cycle_size=48)
    # Total mass 49 over ceil(50/48)=2 cycles -> 24.5 for EVERY cycle,
    # including the tail; a lone 0.5-mass tail is scaled by 24.5, not by 0.5.
    assert denominator == pytest.approx(24.5)


def test_balanced_cycles_avoid_short_tails() -> None:
    from training.ppo import _balanced_micro_batch_cycles

    micro_batches = [[object()] for _ in range(97)]
    cycles = _balanced_micro_batch_cycles(micro_batches[:50], cycle_size=48)
    # 50 micro-batches over 2 optimizer steps -> 25/25, never 48/2: Adam is
    # ~scale-invariant, so a 2-sample cycle would get a full-size update.
    assert [len(cycle) for cycle in cycles] == [25, 25]
    assert _balanced_micro_batch_cycles(micro_batches[:48], cycle_size=48) and [
        len(c) for c in _balanced_micro_batch_cycles(micro_batches[:48], cycle_size=48)
    ] == [48]
    assert [
        len(c) for c in _balanced_micro_batch_cycles(micro_batches, cycle_size=48)
    ] == [33, 32, 32]


@pytest.mark.parametrize("episode_packing", [False, True])
def test_filtering_partial_zero_episode_preserves_original_mass(
    monkeypatch, tmp_path, episode_packing
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.episode_packing = episode_packing
    trainer.config.gradient_accumulation_steps = 4
    first, zero, second = [_tiny_action_sample(advantage=a) for a in (3.0, 0.0, 2.0)]
    zero.step_index = 1
    zero.segment_start = True
    second.trajectory_id = "second"
    monkeypatch.setattr(
        "training.ppo.build_samples", lambda *a, **_kw: [first, zero, second]
    )
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter/runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kw: output_dir / "state.pt",
    )
    contributions = []
    original = trainer._micro_batch_forward_backward

    def record(chunk, *, loss_denominator, **kwargs):
        contributions.extend(
            s.sequence_weight * s.advantage / loss_denominator for s in chunk
        )
        return original(chunk, loss_denominator=loss_denominator, **kwargs)

    monkeypatch.setattr(trainer, "_micro_batch_forward_backward", record)
    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )
    assert result.zero_signal_denominator_scale == pytest.approx(3 / 2)
    assert sum(contributions) == pytest.approx((3.0 + 2.0) / 3.0)


def test_parity_check_pool_selects_newest_policy_densest_first() -> None:
    from training.ppo import _parity_check_pool

    def sample(name: str, policy: int, completion_tokens: int) -> TokenizedSample:
        total = completion_tokens + 2
        return TokenizedSample(
            trajectory_id=name,
            step_index=0,
            input_ids=list(range(total)),
            labels=[-100] * 2 + list(range(2, total)),
            attention_mask=[1] * total,
            completion_mask=[0] * 2 + [1] * completion_tokens,
            reward=0.0,
            return_to_go=0.0,
            advantage=1.0,
            policy_iteration=policy,
        )

    stale_big = sample("stale-big", 17, 500)
    fresh_small = sample("fresh-small", 20, 4)
    fresh_big = sample("fresh-big", 20, 300)

    pool = _parity_check_pool([stale_big, fresh_small, fresh_big])

    # Stale-policy samples are excluded (their captured logprobs differ from
    # current weights by legitimate drift); densest fresh sample leads so the
    # minimum-evidence loop needs the fewest forwards.
    assert [s.trajectory_id for s in pool] == ["fresh-big", "fresh-small"]

    # Unknown policy iterations (windowed path): full pool, density-ordered.
    unknown = [sample("a", -1, 10), sample("b", -1, 50)]
    assert [s.trajectory_id for s in _parity_check_pool(unknown)] == ["b", "a"]


def test_parity_check_accumulates_minimum_evidence_tokens(monkeypatch) -> None:
    # A single tiny first batch must NOT decide the verdict: the check keeps
    # consuming batches until _PARITY_MIN_CHECK_TOKENS completion tokens
    # contributed (bounded by _PARITY_MAX_CHECK_BATCHES).
    import training.ppo as ppo_module

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = TrainingConfig(
        model_name="stub",
        output_dir="unused",
        device="cpu",
        seed=0,
        max_sequence_tokens=4096,
        micro_batch_size=1,
        max_micro_batch_tokens=4096,
        gradient_accumulation_steps=1,
        epochs_per_window=1,
        learning_rate=1e-5,
        weight_decay=0.0,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        max_grad_norm=1.0,
        disable_dropout=True,
        bf16=False,
        gradient_checkpointing=False,
        lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj"]),
        warmup_steps=0,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = type("_Tokenizer", (), {"pad_token_id": 0})()
    trainer._set_policy_mode_for_training_forward = lambda: None  # type: ignore[method-assign]
    trainer._token_logprobs = lambda batch, **_kwargs: batch[  # type: ignore[method-assign]
        "old_token_logprobs"
    ].clone()
    monkeypatch.setattr(ppo_module, "_PARITY_MIN_CHECK_TOKENS", 40)

    def sample(name: str, completion_tokens: int) -> TokenizedSample:
        total = completion_tokens + 2
        return TokenizedSample(
            trajectory_id=name,
            step_index=0,
            input_ids=list(range(1, total + 1)),
            labels=[-100] * 2 + list(range(3, total + 1)),
            attention_mask=[1] * total,
            completion_mask=[0] * 2 + [1] * completion_tokens,
            reward=0.0,
            return_to_go=0.0,
            advantage=1.0,
            old_token_logprobs=[-0.5] * (total - 1),
        )

    # Density ordering puts 30 first; one batch (30 tokens) < 40 minimum, so
    # the check must consume further batches until >= 40.
    samples = [sample("s30", 30), sample("s8", 8), sample("s4", 4)]

    result = trainer._check_rollout_logprob_parity(samples)

    assert result["token_count"] == 40  # 30 + 8 + 2; final prefix is right-truncated
    assert result["mean_abs"] == pytest.approx(0.0)


def test_runtime_lora_rewrite_handles_multimodal_trainer_names() -> None:
    from training.runtime_lora import _rewrite_qwen3_tensor_name

    # Text-only trainer names GAIN the language_model prefix (vLLM layout)...
    assert (
        _rewrite_qwen3_tensor_name(
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
        )
        == "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    )
    # ...and multimodal-trainer names that already carry it pass through
    # unchanged (the crash mode of smoke 3, 2026-07-18).
    already = (
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    )
    assert _rewrite_qwen3_tensor_name(already) == already


def test_historical_dry_optimizer_steps_do_not_consume_the_lr_ramp(monkeypatch) -> None:
    """Historical dry steps remain excluded from the LR ramp on resume."""
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    trainer.config.warmup_steps = 4
    # Ten dry steps happened; the first REAL step is global step 11 and must
    # sit at the bottom of the ramp, not at full rate.
    trainer._dry_optimizer_step_count = 10
    assert trainer._learning_rate_for_step(11) == pytest.approx(0.05 * 1 / 4)
    assert trainer._learning_rate_for_step(12) == pytest.approx(0.05 * 2 / 4)
    assert trainer._learning_rate_for_step(14) == pytest.approx(0.05)
    assert trainer._learning_rate_for_step(40) == pytest.approx(0.05)


def test_previous_policy_kl_zero_at_equality_and_positive_after_drift() -> None:
    import math

    from training.dataset import collate_samples

    trainer = _tiny_loss_test_trainer()
    trainer.config.previous_policy_kl_coef = 0.02
    sample = _weighted_loss_sample("ep", advantage=1.0, weight=1.0)
    batch = collate_samples([sample], pad_token_id=0)
    entropies = torch.zeros((1, 3), dtype=torch.float32)

    matching = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
    matching_terms = trainer._span_loss_terms(batch, matching, entropies)
    # k3 estimator: exactly zero (value AND gradient) when policies match.
    assert matching_terms["previous_policy_kl_loss"].item() == pytest.approx(0.0)

    drift = torch.full((1, 3), math.log(2.0), dtype=torch.float32, requires_grad=True)
    drift_terms = trainer._span_loss_terms(batch, drift, entropies)
    # k3 at log-ratio r: exp(r) - 1 - r = 2 - 1 - log(2).
    assert drift_terms["span_previous_policy_kl"].mean().item() == pytest.approx(
        1.0 - math.log(2.0)
    )
    assert drift_terms["previous_policy_kl_loss"].item() == pytest.approx(
        0.02 * (1.0 - math.log(2.0))
    )


def test_previous_policy_kl_joins_the_training_loss() -> None:
    import math

    trainer = _tiny_loss_test_trainer()
    trainer.config.previous_policy_kl_coef = 0.02
    sample = _weighted_loss_sample("ep", advantage=1.0, weight=1.0)
    metrics = _run_loss_only_micro_batch(
        trainer, [sample], new_logprob_value=math.log(2.0)
    )
    kl = 1.0 - math.log(2.0)
    assert metrics["previous_policy_kl"] == pytest.approx(kl)
    # ratio 2.0 with advantage 1.0 clips at 1.2 -> policy loss -1.2; the KL
    # anchor adds coef * k3 on top.
    assert metrics["loss"] == pytest.approx(-1.2 + 0.02 * kl)


def test_window_trust_region_rejects_restores_and_backs_off() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.max_window_clip_fraction = 0.25
    parameters = trainer._trainable_parameters()
    snapshot = trainer._snapshot_update_state()
    reference = [parameter.detach().clone() for parameter in parameters]

    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(1.0)

    rejected = trainer._apply_window_trust_region(
        snapshot,
        post_update_clip_fraction=0.5,
        iteration_index=3,
    )
    assert rejected is True
    for parameter, expected in zip(parameters, reference):
        assert torch.equal(parameter.detach(), expected)
    assert trainer._trust_region_lr_scale == pytest.approx(0.5)
    assert trainer._consecutive_window_rejections == 1
    # The backoff scales the effective learning rate for the next window.
    assert trainer._learning_rate_for_step(1) == pytest.approx(
        trainer.config.learning_rate * 0.5
    )

    # An accepted window clears the rejection streak and recovers the scale.
    accepted_snapshot = trainer._snapshot_update_state()
    rejected = trainer._apply_window_trust_region(
        accepted_snapshot,
        post_update_clip_fraction=0.05,
        iteration_index=4,
    )
    assert rejected is False
    assert trainer._consecutive_window_rejections == 0
    assert trainer._trust_region_lr_scale == pytest.approx(0.625)


def test_window_trust_region_disabled_accepts_but_enabled_requires_probe() -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.max_window_clip_fraction = 0.0
    assert (
        trainer._apply_window_trust_region(
            None,
            post_update_clip_fraction=1.0,
            iteration_index=0,
        )
        is False
    )
    trainer.config.max_window_clip_fraction = 0.25
    snapshot = trainer._snapshot_update_state()
    with pytest.raises(RuntimeError, match="requires a valid update probe"):
        trainer._apply_window_trust_region(
            snapshot,
            post_update_clip_fraction=None,
            iteration_index=0,
        )


def test_window_trust_region_aborts_after_consecutive_rejections() -> None:
    from training.ppo import _TRUST_REGION_MAX_CONSECUTIVE_REJECTIONS

    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.max_window_clip_fraction = 0.25
    trainer._consecutive_window_rejections = (
        _TRUST_REGION_MAX_CONSECUTIVE_REJECTIONS - 1
    )
    snapshot = trainer._snapshot_update_state()
    with pytest.raises(RuntimeError, match="consecutive"):
        trainer._apply_window_trust_region(
            snapshot,
            post_update_clip_fraction=0.5,
            iteration_index=9,
        )


@pytest.mark.parametrize(("learning_rate", "rejected"), [(1e-5, False), (0.25, True)])
def test_train_window_reports_trust_region_fields(
    monkeypatch, tmp_path, learning_rate, rejected
) -> None:
    trainer = _make_tiny_ppo_trainer(advantage=1.0, learning_rate=learning_rate)
    trainer.config.max_window_clip_fraction = 0.25
    sample = _tiny_action_sample(advantage=1.0)
    monkeypatch.setattr("training.ppo.build_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda output_dir, iteration_index: (
            output_dir / "adapter",
            output_dir / "adapter" / "runtime_vllm",
            0.0,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_save_training_state",
        lambda output_dir, **_kwargs: output_dir / "trainer_state_iter_000000.pt",
    )

    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )

    # Even lag-zero, one-cycle windows are checked AFTER their first step.
    assert result.window_update_rejected is rejected
    assert result.trust_region_lr_scale == pytest.approx(0.5 if rejected else 1.0)
    assert result.mean_previous_policy_kl >= 0.0


def test_entropy_control_treats_rejected_window_as_no_signal() -> None:
    from types import SimpleNamespace

    from training.cli import (
        _initial_entropy_control_state,
        _update_entropy_control_state,
    )
    from training.config import EntropyControlConfig

    config = SimpleNamespace(
        training=SimpleNamespace(
            entropy_control=EntropyControlConfig.from_dict({"enabled": True})
        )
    )
    state = _initial_entropy_control_state(config, None)
    coef_before = state["current_coef"]
    # A trust-region-rejected window rolls its update back but still reports
    # positive optimizer_steps and the entropy measured during the discarded
    # update; the controller must not integrate that phantom signal.
    updated = _update_entropy_control_state(
        config=config,
        state=state,
        train_result=SimpleNamespace(
            mean_entropy=0.01,
            sample_count=64,
            optimizer_steps=8,
            window_update_rejected=True,
        ),
    )
    assert updated["last_action"] == "no_signal"
    assert updated["current_coef"] == pytest.approx(coef_before)
    assert updated["last_mean_entropy"] is None

    # The same result with the window accepted DOES step the coefficient up.
    accepted = _update_entropy_control_state(
        config=config,
        state=state,
        train_result=SimpleNamespace(
            mean_entropy=0.01,
            sample_count=64,
            optimizer_steps=8,
            window_update_rejected=False,
        ),
    )
    assert accepted["last_action"] == "step_up"
    assert accepted["current_coef"] > coef_before


def _packed_two_span_sample() -> TokenizedSample:
    # prompt [1, 2]; span 1 = token 3 (1 token, +3.0); span 2 = tokens 4-6
    # (3 tokens, -1.5). Target-aligned span ids/advantages (5 targets).
    return TokenizedSample(
        trajectory_id="ep",
        step_index=0,
        input_ids=[1, 2, 3, 4, 5, 6],
        labels=[-100, -100, 3, 4, 5, 6],
        attention_mask=[1] * 6,
        completion_mask=[0, 0, 1, 1, 1, 1],
        reward=0.0,
        return_to_go=0.0,
        advantage=3.0,
        prompt_token_count=2,
        old_token_logprobs=[0.0] * 5,
        span_ids=[0, 1, 2, 2, 2],
        token_advantages=[0.0, 3.0, -1.5, -1.5, -1.5],
        span_count=2,
        sequence_weight=1.0,
    )


def test_span_mean_aggregation_weights_spans_equally() -> None:
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "span_mean"
    metrics = _run_loss_only_micro_batch(trainer, [_packed_two_span_sample()])
    assert metrics["loss"] == pytest.approx(-(0.5 * 3.0 + 0.5 * -1.5))


def test_token_mean_aggregation_weights_spans_by_token_count() -> None:
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "token_mean"
    metrics = _run_loss_only_micro_batch(trainer, [_packed_two_span_sample()])
    # 1 of 4 completion tokens carries +3.0, 3 of 4 carry -1.5.
    assert metrics["loss"] == pytest.approx(-(0.25 * 3.0 + 0.75 * -1.5))


def test_token_mean_preserves_explicit_sequence_mass() -> None:
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "token_mean"
    chunk = [
        _weighted_loss_sample("ep", advantage=3.0, weight=1.0 / 3.0),
        _weighted_loss_sample("ep", advantage=-1.5, weight=2.0 / 3.0),
    ]
    metrics = _run_loss_only_micro_batch(trainer, chunk)
    # Single-span sequences: the sequence weights alone decide the mass.
    assert metrics["loss"] == pytest.approx(-(1.0 / 3.0 * 3.0 + 2.0 / 3.0 * -1.5))


def test_episode_mean_aggregation_weights_spans_by_token_count_within_the_episode() -> None:
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "episode_mean"
    metrics = _run_loss_only_micro_batch(trainer, [_packed_two_span_sample()])
    # Within a sequence the spread is token-proportional, as for token_mean:
    # 1 of 4 completion tokens carries +3.0, 3 of 4 carry -1.5.
    assert metrics["loss"] == pytest.approx(-(0.25 * 3.0 + 0.75 * -1.5))


def test_episode_mean_gives_each_episode_unit_mass_over_its_tokens() -> None:
    from training.episode_packing import assign_loss_weights

    short = _weighted_loss_sample("short", advantage=2.0, weight=1.0)  # 2 completion tokens
    long_a = _weighted_loss_sample("long", advantage=3.0, weight=1.0)  # 2 tokens
    long_b = _weighted_loss_sample("long", advantage=-1.5, weight=1.0)  # 2 tokens
    samples = [short, long_a, long_b]
    assign_loss_weights(samples, loss_aggregation="episode_mean")
    # Every episode sums to one; a sample's share is its token share.
    assert short.sequence_weight == pytest.approx(1.0)
    assert long_a.sequence_weight == pytest.approx(0.5)
    assert long_b.sequence_weight == pytest.approx(0.5)
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = "episode_mean"
    metrics = _run_loss_only_micro_batch(trainer, samples)
    # Each token of the short episode carries 1/2; each token of the long
    # episode carries 1/4, so the 2-token short sample matches the whole
    # 4-token long episode in mass.
    assert metrics["loss"] == pytest.approx(-(1.0 * 2.0 + 0.5 * 3.0 + 0.5 * -1.5))


def test_loss_aggregation_accepts_episode_mean() -> None:
    config = TrainingConfig.from_dict(
        {"model_name": "m", "output_dir": "o", "loss_aggregation": "episode_mean"}
    )
    assert config.loss_aggregation == "episode_mean"


@pytest.mark.parametrize("aggregation", ["token_mean", "span_mean", "episode_mean"])
def test_compaction_preserves_episode_loss_and_token_gradients(
    aggregation: str,
) -> None:
    from dataclasses import replace
    from training.dataset import collate_samples
    from training.episode_packing import pack_episode_samples

    first = _weighted_loss_sample("episode", advantage=3.0, weight=1.0)
    completion = [6, 7, 8, 9, 10, 11]
    prompt = first.input_ids + [5]
    second = replace(
        first,
        step_index=1,
        input_ids=prompt + completion,
        labels=[-100] * len(prompt) + completion,
        attention_mask=[1] * (len(prompt) + len(completion)),
        completion_mask=[0] * len(prompt) + [1] * len(completion),
        prompt_token_count=len(prompt),
        old_token_logprobs=[0.0] * 10,
        advantage=-1.5,
    )
    compacted = replace(
        second,
        input_ids=[20, 21] + completion,
        labels=[-100, -100] + completion,
        attention_mask=[1] * 8,
        completion_mask=[0, 0] + [1] * 6,
        prompt_token_count=2,
        old_token_logprobs=[0.0] * 7,
        segment_start=True,
    )
    trainer = _tiny_loss_test_trainer()
    trainer.config.loss_aggregation = aggregation
    losses, gradients = [], []
    for samples in ([first, second], [first, compacted]):
        packed, _ = pack_episode_samples(
            samples, max_sequence_tokens=32, loss_aggregation=aggregation
        )
        batch = collate_samples(packed, pad_token_id=0)
        logprobs = torch.zeros_like(batch["old_token_logprobs"], requires_grad=True)
        terms = trainer._span_loss_terms(batch, logprobs, torch.zeros_like(logprobs))
        terms["policy_loss"].backward()
        losses.append(terms["policy_loss"].item())
        gradients.append(logprobs.grad[batch["span_ids"] > 0])
    assert losses[1] == pytest.approx(losses[0])
    torch.testing.assert_close(gradients[1], gradients[0])


def test_loss_aggregation_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="loss_aggregation"):
        TrainingConfig.from_dict(
            {"model_name": "m", "output_dir": "o", "loss_aggregation": "bogus"}
        )


def test_rollout_metrics_report_rejected_attempts() -> None:
    from training.contracts import RolloutStep, RolloutTrajectory
    from training.metrics import summarize_rollouts

    def step(index: int, *, void: bool = False) -> RolloutStep:
        return RolloutStep(
            step_index=index,
            prompt_messages=[],
            tool_schemas=[],
            assistant_message={},
            reward=0.0,
            done=False,
            prompt_token_ids=[1],
            completion_token_ids=[2, 3],
            metadata={"void_attempt": True} if void else {},
        )

    rollout = RolloutTrajectory(
        trajectory_id="t",
        task_id="g",
        steps=[step(0), step(1), step(2, void=True)],
        done=True,
        trajectory_reward=0.0,
        metadata={
            "void_attempts": 1,
            "levels_completed": 0,
            "game_total_levels": 1,
            "run_total_actions": 2,
        },
    )
    metrics = summarize_rollouts([rollout])
    # Generation-round stats cover reported turns only.
    assert metrics.avg_generation_rounds_per_rollout == 2
    assert metrics.avg_tokens_per_generation == pytest.approx(2.0)
    assert metrics.void_attempt_fraction == pytest.approx(1.0 / 3.0)
    assert "rollout/void_attempt_fraction" in metrics.as_scalars()


def test_inference_config_parses_malformed_reply_knobs() -> None:
    from training.config import InferenceConfig

    base = {
        "base_url": "http://x/v1",
        "model_id": "m",
        "games": ["g"],
        "max_runtime_minutes": 1,
        "agent_max_completion_tokens": 4096,
    }
    # Training defaults: every draw the policy makes is a step.
    cfg = InferenceConfig.from_dict(dict(base))
    assert cfg.agent_malformed_retries == 0
    assert cfg.agent_accept_malformed_output is True
    cfg = InferenceConfig.from_dict(
        {**base, "agent_malformed_retries": 5, "agent_accept_malformed_output": False}
    )
    assert cfg.agent_malformed_retries == 5
    assert cfg.agent_accept_malformed_output is False
    # An empty retained reply keeps the nudge loop unless asked to end the episode.
    assert InferenceConfig.from_dict(dict(base)).agent_terminate_on_empty_reply is False
    cfg = InferenceConfig.from_dict({**base, "agent_terminate_on_empty_reply": True})
    assert cfg.agent_terminate_on_empty_reply is True


def test_reference_kl_anchor_adds_k3_term_against_the_base_policy() -> None:
    trainer = _tiny_loss_test_trainer()
    trainer.config.reference_kl_coef = 0.5
    # Base policy sits 1 nat below the current policy on every token.
    trainer._reference_token_logprobs = lambda batch: torch.full(
        (batch["input_ids"].shape[0], batch["input_ids"].shape[1] - 1), -1.0
    )
    sample = _weighted_loss_sample("anchored", advantage=0.0, weight=1.0)
    metrics = _run_loss_only_micro_batch(trainer, [sample])
    k3 = math.exp(-1.0) - 1.0 - (-1.0)
    assert metrics["reference_kl"] == pytest.approx(k3)
    assert metrics["loss"] == pytest.approx(0.5 * k3)


def test_reference_kl_is_off_by_default() -> None:
    trainer = _tiny_loss_test_trainer()

    def _forbidden(batch):
        raise AssertionError("reference forward must not run when the coefficient is 0")

    trainer._reference_token_logprobs = _forbidden
    sample = _weighted_loss_sample("plain", advantage=1.0, weight=1.0)
    metrics = _run_loss_only_micro_batch(trainer, [sample])
    assert metrics["reference_kl"] == 0.0
    assert metrics["loss"] == pytest.approx(-1.0)


def test_collapse_watch_reads_the_leading_indicators() -> None:
    from types import SimpleNamespace

    from training.cli import _collapse_alarm_reasons, _format_break_fraction

    rollout = SimpleNamespace(
        steps=[
            SimpleNamespace(metadata={"format_break": False}),
            SimpleNamespace(metadata={"format_break": True}),
            SimpleNamespace(metadata={"format_break": True, "void_attempt": True}),
        ]
    )
    assert _format_break_fraction([rollout]) == pytest.approx(0.5)
    healthy = _collapse_alarm_reasons(
        format_break_fraction=0.005,
        format_break_fraction_alarm=0.02,
        actions_per_rollout=120.0,
        actions_history=[100.0, 130.0, 120.0],
        actions_drop_fraction=0.5,
        watch_windows=3,
    )
    assert healthy == []
    reasons = _collapse_alarm_reasons(
        format_break_fraction=0.05,
        format_break_fraction_alarm=0.02,
        actions_per_rollout=30.0,
        actions_history=[130.0, 120.0, 30.0],
        actions_drop_fraction=0.5,
        watch_windows=3,
    )
    assert len(reasons) == 2
    # Too little history: the actions rule stays quiet, the fraction rule fires.
    early = _collapse_alarm_reasons(
        format_break_fraction=0.05,
        format_break_fraction_alarm=0.02,
        actions_per_rollout=30.0,
        actions_history=[130.0],
        actions_drop_fraction=0.5,
        watch_windows=3,
    )
    assert len(early) == 1
    # One spiky window does not set the bar: the median of the trailing windows
    # is the baseline, so a steady 30 after a single 68 stays healthy.
    spiky = _collapse_alarm_reasons(
        format_break_fraction=0.0,
        format_break_fraction_alarm=0.02,
        actions_per_rollout=30.0,
        actions_history=[20.0, 25.0, 68.0, 52.0, 36.0, 30.0, 30.0, 30.0],
        actions_drop_fraction=0.5,
        watch_windows=8,
    )
    assert spiky == []


def test_initial_runtime_lora_export_is_loaded_without_changing_the_policy(
    monkeypatch, tmp_path
):
    from training import cli, official_eval

    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    before = {
        name: value.detach().clone() for name, value in trainer.model.named_parameters()
    }
    initial_steps = trainer.optimizer_step_count
    runtime = cli._publish_initial_policy(tmp_path, trainer, resume_metadata={})
    assert official_eval._await_published_policy(tmp_path) == (-1, runtime)
    assert runtime_lora_has_effective_delta(runtime) is False
    assert (runtime / "adapter_model.safetensors").is_file()
    assert trainer.optimizer_step_count == initial_steps
    assert all(
        torch.equal(value, before[name])
        for name, value in trainer.model.named_parameters()
    )
    initial_tensors = _load_adapter_tensors(runtime)
    assert any("lora_A" in name for name in initial_tensors)
    assert any("lora_B" in name for name in initial_tensors)


def test_reexporting_initial_eval_does_not_rewind_latest_runtime_or_policy(
    monkeypatch, tmp_path
):
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    initial, _, _ = trainer._export_policy_snapshot(tmp_path, -1)
    _, current_runtime, _ = trainer._export_policy_snapshot(tmp_path, 3)
    latest = tmp_path / "latest_runtime_adapter"
    assert latest.resolve() == current_runtime.resolve()
    before = {
        name: value.detach().clone() for name, value in trainer.model.named_parameters()
    }
    trainer._save_runtime_adapter(initial, publish_latest=False)
    assert latest.resolve() == current_runtime.resolve()
    assert all(
        torch.equal(value, before[name])
        for name, value in trainer.model.named_parameters()
    )
    assert (initial / "PINNED").exists()


def _test_state_critic(seed: int = 123) -> StateCriticBaseline:
    return StateCriticBaseline(
        game_embedding_buckets=32,
        game_embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.01,
        seed=seed,
    )
