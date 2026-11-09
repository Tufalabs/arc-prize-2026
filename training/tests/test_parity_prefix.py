"""The safety spot-check preserves causal context without scoring whole episodes."""

import copy
from types import SimpleNamespace

import pytest
import torch

from training.dataset import TokenizedSample
from training.ppo import PPOTrainer, _parity_check_pool, _parity_prefix


def sample(name, prompt_tokens, output_tokens, *, step_index=0, segment_start=False):
    ids = list(range(1, prompt_tokens + output_tokens + 1))
    return TokenizedSample(
        trajectory_id=name,
        step_index=step_index,
        input_ids=ids,
        labels=[-100] * prompt_tokens + ids[prompt_tokens:],
        attention_mask=[1] * len(ids),
        completion_mask=[0] * prompt_tokens + [1] * output_tokens,
        old_token_logprobs=[0.0] * (prompt_tokens - 1) + [-0.4] * output_tokens,
        prompt_token_count=prompt_tokens,
        reward=1.0,
        return_to_go=1.0,
        advantage=1.0,
        segment_start=segment_start,
    )


def trainer(forwards):
    result = PPOTrainer.__new__(PPOTrainer)
    result.config = SimpleNamespace(
        micro_batch_size=1,
        max_micro_batch_tokens=65536,
        rollout_logprob_check_batches=1,
    )
    result.device = torch.device("cpu")
    result.tokenizer = SimpleNamespace(pad_token_id=0)
    result._set_policy_mode_for_training_forward = lambda: None

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        return batch["old_token_logprobs"].clone()

    result._token_logprobs = forward
    return result


def test_parity_right_truncation_preserves_every_causal_token_and_original_sample():
    original = sample("long", 7000, 20000)
    before = copy.deepcopy(original)
    prefix = _parity_prefix(original, 128)
    assert prefix.input_ids == original.input_ids[:7128]
    assert prefix.old_token_logprobs == original.old_token_logprobs[:7127]
    assert prefix.completion_mask == [0] * 7000 + [1] * 128
    assert original == before


def test_parity_reset_phase_prefers_reset_over_start_or_late_context():
    late = sample("late", 30000, 20000, step_index=20)
    start = sample("start", 600, 1000)
    reset = sample("reset", 400, 1000, step_index=10, segment_start=True)
    assert [
        s.trajectory_id
        for s in _parity_check_pool([late, start, reset], policy_iteration=0)
    ] == [
        "reset",
        "start",
        "late",
    ]
    forwards = []
    metrics = trainer(forwards)._check_rollout_logprob_parity(
        [late, start, reset], trainer_policy_iteration=0
    )
    assert metrics["token_count"] == 128
    assert metrics["input_token_count"] == 528
    assert metrics["sample_count"] == metrics["batch_count"] == 1
    assert forwards[0].tolist() == [reset.input_ids[:528]]


def test_parity_accumulates_short_draws_without_overshooting_output_budget():
    forwards = []
    metrics = trainer(forwards)._check_rollout_logprob_parity(
        [sample("a", 4, 70), sample("b", 5, 70)]
    )
    assert metrics["token_count"] == 128
    assert [batch.shape[1] for batch in forwards] == [74, 63]


def test_selected_sampled_logprob_mismatch_remains_fatal():
    model = trainer([])
    model._token_logprobs = lambda batch: batch["old_token_logprobs"] + 0.5
    metrics = model._check_rollout_logprob_parity([sample("bad", 400, 1000)])
    assert "outside tolerances" in model._rollout_logprob_parity_failure(
        metrics, sample_count=1
    )


def test_non_finite_rescored_tokens_are_fatal():
    forwards = []
    model = trainer(forwards)

    def score(batch):
        forwards.append(batch["input_ids"].clone())
        scores = batch["old_token_logprobs"].clone()
        if len(forwards) == 1:
            scores[0, -1] = float("nan")
        return scores

    model._token_logprobs = score
    with pytest.raises(RuntimeError, match="1 non-finite completion token value"):
        model._check_rollout_logprob_parity([sample("nan", 4, 70), sample("ok", 5, 70)])
    # Every selected forward ran before the verdict (ranks stay in lockstep).
    assert len(forwards) == 2


def test_long_only_context_is_preserved_even_if_check_is_expensive():
    forwards = []
    model = trainer(forwards)
    metrics = model._check_rollout_logprob_parity(
        [sample("long-only", 32000, 20000, step_index=15)]
    )
    assert metrics["token_count"] == 128
    assert forwards[0].shape[1] == 32128
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""


def test_no_completion_evidence_does_not_pass():
    model = trainer([])
    metrics = model._check_rollout_logprob_parity([sample("no-target", 4, 0)])
    with pytest.raises(RuntimeError, match="did not compare any completion tokens"):
        model._rollout_logprob_parity_failure(metrics, sample_count=1)


@pytest.mark.parametrize("available,passes", [(1500, True), (200, False)])
def test_rare_outlier_expansion_retains_original_token_and_counts_all_work(
    available, passes
):
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        # The same original sampled token stays discrepant on the wider pass.
        return batch["old_token_logprobs"] + (batch["input_ids"][:, 1:] == 5) * 2.01

    model._token_logprobs = forward
    metrics = model._check_rollout_logprob_parity([sample("tail", 4, available)])
    assert metrics["expanded"] is True
    assert metrics["token_count"] == min(1000, available)
    assert metrics["outlier_count"] == 1
    assert len(forwards) == 2
    assert forwards[1][0, : forwards[0].shape[1]].tolist() == forwards[0][0].tolist()
    assert metrics["input_token_count"] == sum(batch.numel() for batch in forwards)
    assert metrics["sample_count"] == metrics["batch_count"] == 2
    assert (
        model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""
    ) == passes


def test_outlier_that_alone_lifts_the_spot_check_mean_still_expands():
    # 7 nats over 128 targets lifts the mean to ~0.055 with every other
    # token exact, so only the outlier stands between the check and a pass.
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        return batch["old_token_logprobs"] + (batch["input_ids"][:, 1:] == 5) * 7.0

    model._token_logprobs = forward
    metrics = model._check_rollout_logprob_parity([sample("tail", 4, 1500)])
    assert metrics["expanded"] is True
    assert metrics["token_count"] == 1000
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""


def test_systematic_mismatch_expands_once_and_still_fails():
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"])
        return batch["old_token_logprobs"] + 0.5

    model._token_logprobs = forward
    metrics = model._check_rollout_logprob_parity([sample("bad", 4, 3000)])
    # Every failure gathers the 1,000-target evidence once; a systematic
    # offset does not shrink with more tokens, so the gate still fails.
    assert metrics["expanded"] is True
    assert metrics["token_count"] >= 1000
    assert len(forwards) == 2
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1)


@pytest.mark.parametrize("available", [1500, 200])
@pytest.mark.parametrize("tail_offset,passes", [(0.0, True), (0.055, False)])
def test_marginal_mean_expansion_uses_available_evidence_and_counts_all_work(
    available, tail_offset, passes, monkeypatch, caplog
):
    forwards = []
    model = trainer(forwards)
    clock = iter([1.0, 3.0, 4.0, 7.0])
    monkeypatch.setattr("training.ppo.time.perf_counter", lambda: next(clock))

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        offsets = torch.where(batch["input_ids"][:, 1:] <= 132, 0.055, tail_offset)
        return batch["old_token_logprobs"] + offsets

    model._token_logprobs = forward
    with caplog.at_level("INFO", logger="training.ppo"):
        metrics = model._check_rollout_logprob_parity(
            [sample("marginal", 4, available)]
        )
    assert metrics["expanded"] is True
    assert metrics["token_count"] == min(1000, available)
    assert len(forwards) == 2
    assert forwards[0].shape[1] == 132
    assert forwards[1][0, :132].tolist() == forwards[0][0].tolist()
    assert metrics["mean_abs"] == pytest.approx(
        (128 * 0.055 + (metrics["token_count"] - 128) * tail_offset)
        / metrics["token_count"]
    )
    assert metrics["input_token_count"] == sum(batch.numel() for batch in forwards)
    assert metrics["sample_count"] == metrics["batch_count"] == 2
    assert metrics["forward_seconds"] == 5.0
    assert "for marginal failure: mean_abs" in caplog.text
    assert (
        model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""
    ) == passes


@pytest.mark.parametrize("base_offset", [0.0, 0.04])
def test_marginal_p95_failure_expands(base_offset, caplog):
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        # Eight discrepant tokens put the 128-token p95 over tolerance,
        # but constitute less than 5% of the expanded evidence.
        return (
            batch["old_token_logprobs"]
            + base_offset
            + (batch["input_ids"][:, 1:] <= 12) * 0.26
        )

    model._token_logprobs = forward
    with caplog.at_level("INFO", logger="training.ppo"):
        metrics = model._check_rollout_logprob_parity([sample("p95", 4, 1500)])
    assert metrics["expanded"] is True
    assert metrics["token_count"] == 1000
    assert [batch.shape[1] for batch in forwards] == [132, 1004]
    assert metrics["mean_abs"] == pytest.approx(base_offset + 8 * 0.26 / 1000)
    assert metrics["p95_abs"] == pytest.approx(base_offset)
    trigger = "mean_abs, p95_abs" if base_offset else "p95_abs"
    assert f"for marginal failure: {trigger}" in caplog.text
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""


@pytest.mark.parametrize("statistic", ["mean", "p95"])
def test_failure_beyond_marginal_band_expands_and_still_fails(statistic, caplog):
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        # A uniform 0.08 shift fails the mean; 0.4 on every tenth token fails
        # the p95 at 128 targets and still at 1,000, unlike a mismatch confined
        # to the first few tokens, which more evidence rightly dilutes.
        offsets = (
            0.08 if statistic == "mean" else (batch["input_ids"][:, 1:] % 10 == 0) * 0.4
        )
        return batch["old_token_logprobs"] + offsets

    model._token_logprobs = forward
    with caplog.at_level("INFO"):
        metrics = model._check_rollout_logprob_parity([sample("systematic", 4, 1500)])
    assert metrics["expanded"] is True
    assert metrics["token_count"] >= 1000
    assert len(forwards) == 2
    assert "for spot-check failure:" in caplog.text
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1)


def test_marginal_mean_with_outliers_uses_outlier_driven_target():
    forwards = []
    model = trainer(forwards)

    def forward(batch):
        forwards.append(batch["input_ids"].clone())
        token_ids = batch["input_ids"][:, 1:]
        return (
            batch["old_token_logprobs"]
            + (token_ids <= 132) * 0.055
            + (token_ids <= 6) * 2.01
        )

    model._token_logprobs = forward
    metrics = model._check_rollout_logprob_parity([sample("mixed", 4, 3000)])
    assert metrics["expanded"] is True
    assert metrics["token_count"] == 2000
    assert metrics["outlier_count"] == 2
    assert [batch.shape[1] for batch in forwards] == [132, 2004]
    assert model._rollout_logprob_parity_failure(metrics, sample_count=1) == ""
