"""Exercise parity beyond cheap initial prompts without changing training RNG."""

import random
from types import SimpleNamespace

import pytest
import torch

from training.dataset import TokenizedSample
from training.ppo import PPOTrainer, _parity_check_pool, _parity_prefix


def sample(name, prompt, *, step=0, reset=False, completion=256, policy=2):
    size = prompt + completion
    return TokenizedSample(
        trajectory_id=name,
        step_index=step,
        input_ids=list(range(1, size + 1)),
        labels=[-100] * prompt + list(range(prompt + 1, size + 1)),
        attention_mask=[1] * size,
        completion_mask=[0] * prompt + [1] * completion,
        reward=0.0,
        return_to_go=0.0,
        advantage=1.0,
        old_token_logprobs=[-0.5] * (size - 1),
        prompt_token_count=prompt,
        segment_start=reset,
        policy_iteration=policy,
    )


@pytest.mark.parametrize(
    "policy,expected",
    [
        (-1, "initial"),
        (0, "reset"),
        (1, "initial"),
        (2, "long"),
        (3, "initial"),
        (4, "reset"),
        (5, "initial"),
        (6, "long"),
    ],
)
def test_coverage_rotates_and_prioritizes_complete_evidence(policy, expected):
    samples = [
        sample("initial", 20),
        sample("reset", 40, step=8, reset=True),
        sample("short", 100, step=9),
        sample("long", 1000, step=30),
        sample("tiny-tail", 2000, step=31, completion=1),
        sample("stale-long", 3000, step=35, policy=1),
    ]
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    ordered = _parity_check_pool(samples, policy_iteration=policy)
    assert ordered[0].trajectory_id == expected
    assert "stale-long" not in [s.trajectory_id for s in ordered]
    assert [s.trajectory_id for s in ordered] == [
        s.trajectory_id
        for s in _parity_check_pool(list(reversed(samples)), policy_iteration=policy)
    ]
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize("policy", [0, 2])
def test_missing_context_kind_falls_back_to_available_evidence(policy):
    initial = sample("initial", 20)
    assert _parity_check_pool([initial], policy_iteration=policy) == [initial]


def test_long_prefix_keeps_every_causal_token_and_caps_only_output():
    original = sample("long", 24000, step=30)
    prefix = _parity_prefix(original, 128)
    assert prefix.input_ids == original.input_ids[:24128]
    assert prefix.old_token_logprobs == original.old_token_logprobs[:24127]
    assert sum(prefix.completion_mask) == 128
    assert prefix.completion_mask[:24000] == [0] * 24000
    assert len(original.input_ids) == 24256


@pytest.mark.parametrize("policy,reset", [(0, True), (2, False)])
def test_noninitial_parity_mismatch_is_fatal_and_restores_state(policy, reset):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        micro_batch_size=1,
        rollout_logprob_check_batches=1,
        max_micro_batch_tokens=32768,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.model = torch.nn.Linear(1, 1)
    weights = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    trainer._parity_policy_snapshots = {policy: weights}
    trainer._set_policy_mode_for_training_forward = lambda: None
    seen = []

    def forward(batch):
        seen.append(batch["input_ids"].shape[1])
        return batch["old_token_logprobs"].clone() - 0.5

    trainer._token_logprobs = forward
    samples = [
        sample("initial", 20, policy=policy),
        sample("later", 1000, step=8, reset=reset, policy=policy),
    ]
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        RuntimeError, match="Same-checkpoint logprob parity check failed"
    ):
        trainer._check_window_logprob_parity(
            samples, trainer_policy_iteration=policy, policy_lag_windows=0
        )
    # The 128-target spot check first, then the expansion over the available
    # captures; the systematic offset survives the extra evidence, so it is
    # still fatal and the trainer state is still restored.
    assert seen[0] == 1128 and len(seen) >= 2
    assert torch.equal(torch.get_rng_state(), rng)
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, weights[name])


def biased(name, prompt, *, step=12, completion=256, policy=2, token=248058):
    """A summary draw: sampled under the token ban, rescored with it."""
    result = sample(name, prompt, step=step, completion=completion, policy=policy)
    result.token_logit_bias = {
        token: [0.0] * (prompt - 1) + [-100.0] * completion,
    }
    return result


@pytest.mark.parametrize("policy", [1, 5])
def test_summary_phase_checks_a_biased_draw_first(policy):
    samples = [
        sample("initial", 20),
        sample("reset", 40, step=8, reset=True),
        sample("long", 1000, step=30),
        biased("summary", 3000),
    ]
    ordered = _parity_check_pool(samples, policy_iteration=policy)
    assert ordered[0].trajectory_id == "summary"
    # Other phases keep their own evidence first; the biased draw is never a
    # "continuation" stand-in.
    assert _parity_check_pool(samples, policy_iteration=2)[0].trajectory_id == "long"
    assert _parity_check_pool(samples, policy_iteration=0)[0].trajectory_id == "reset"


def test_biased_parity_targets_are_counted_and_measured_separately():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        micro_batch_size=1,
        rollout_logprob_check_batches=1,
        max_micro_batch_tokens=32768,
    )
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._set_policy_mode_for_training_forward = lambda: None

    def forward(batch):
        # Off by 0.01 on biased targets only: what a sampler/learner bias
        # disagreement would look like, below the gate's tolerance.
        shifted = batch["old_token_logprobs"].clone()
        biased_mask = (batch["logit_bias_values"] != 0).any(dim=-1)
        return shifted - 0.01 * biased_mask

    trainer._token_logprobs = forward
    metrics = trainer._check_rollout_logprob_parity(
        [biased("summary", 300, completion=64)], trainer_policy_iteration=1
    )
    assert metrics["token_count"] == 64
    assert metrics["biased_token_count"] == 64
    assert metrics["biased_mean_abs"] == pytest.approx(0.01, abs=1e-6)
