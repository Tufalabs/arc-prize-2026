"""The served-acceptance estimator's pieces against vLLM's own."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from drafter.loss import _top_k_top_p


def test_top_k_top_p_matches_vllm():
    # Where the package is copied without its engine (arc-training's CI), there is nothing to compare with.
    sampler = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler", reason="Requires vLLM")
    apply_top_k_top_p_pytorch = sampler.apply_top_k_top_p_pytorch

    torch.manual_seed(0)
    logits = torch.randn(16, 5000) * 3
    ours = _top_k_top_p(logits.clone(), 194, 0.8899009607194754)
    theirs = apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((16,), 194), torch.full((16,), 0.8899009607194754), allow_cpu_sync=True
    )
    assert torch.equal(torch.isinf(ours), torch.isinf(theirs))
    torch.testing.assert_close(ours.softmax(-1), theirs.softmax(-1))


def test_acceptance_ratio_averages_to_acceptance():
    """Over the target's own samples, the ratio min(1, q/p) averages to the
    acceptance sum min(p, q): the identity the reach metric rests on."""
    from drafter.loss import acceptance_metrics

    torch.manual_seed(0)
    vocab, width = 300, 32
    head = torch.randn(vocab, width)
    student, teacher = torch.randn(1, width).expand(vocab, width), torch.randn(1, width).expand(vocab, width)
    ratios: dict = {}
    sums = acceptance_metrics(student, teacher, head, torch.arange(vocab), ratios=ratios)
    p = (F.linear(teacher[:1], head) / 0.6).softmax(-1)[0]
    torch.testing.assert_close(float((p * ratios["t0.6"]).sum()), sums["accept_t0.6"] / vocab)


def test_draft_weighting_follows_the_engine():
    """A draft from anchor t keeping j tokens puts the next at t + 1 + j: all
    kept, drafts land every k + 1 tokens; none kept, on every anchor."""
    from drafter.train import _draft_weighted

    k, start, end = 3, 10, 30  # anchors 9..27
    ones = [torch.ones(40) for _ in range(k)]
    sums = _draft_weighted(ones, [(start, end)])
    assert sums["rows"] == len(range(9, 28, k + 1)) and sums["pos2"] == sums["rows"]
    zeros = [torch.zeros(40) for _ in range(k)]
    sums = _draft_weighted(zeros, [(start, end)])
    assert sums["rows"] == 19 and sums["pos0"] == 0.0


def test_weighted_distillation_matches_autograd():
    """Per-row weights scale each row's KL and its gradient; none is
    upstream's plain sum."""
    from drafter.loss import ChunkedDistillation

    torch.manual_seed(0)
    rows, width, vocab, temperature = 7, 16, 50, 0.6
    head, teacher = torch.randn(vocab, width), torch.randn(rows, width)
    hidden = torch.randn(rows, width, requires_grad=True)
    weights = torch.rand(rows)
    targets = torch.randint(vocab, (rows,))

    def reference(w):
        log_t = (F.linear(teacher, head) / temperature).log_softmax(-1)
        log_s = (F.linear(hidden, head) / temperature).log_softmax(-1)
        return ((log_t.exp() * (log_t - log_s)).sum(-1) * w).sum() * temperature**2

    for w in (None, weights):
        ours = ChunkedDistillation.apply(hidden, head, teacher, head, targets, temperature, 0.0, 3, w)
        (grad,) = torch.autograd.grad(ours, hidden)
        expected = reference(torch.ones(rows) if w is None else w)
        (expected_grad,) = torch.autograd.grad(expected, hidden)
        torch.testing.assert_close(ours, expected)
        torch.testing.assert_close(grad, expected_grad)


def test_serving_weights_at_step_zero_sum_to_one():
    """Every anchor a draft can start at is a step-0 row, so step 0's
    weights are a distribution over them; later steps carry the reach."""
    from drafter.train import Trainer

    k, start, end = 3, 10, 30
    torch.manual_seed(0)
    reach = [torch.rand(40)]
    for _ in range(k - 1):
        reach.append(reach[-1] * torch.rand(40))
    picked = [torch.arange(start - 1, end - 2 - j) for j in range(k)]
    weights = Trainer._serving_weights(reach, [(start, end)], picked)
    torch.testing.assert_close(float(weights[0].sum()), 1.0)
    assert float(weights[1].sum()) < 1.0 and float(weights[2].sum()) < float(weights[1].sum())
