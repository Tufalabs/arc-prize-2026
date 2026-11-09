"""Sparse forward KL with an explicit residual bucket and bounded logit memory."""

import torch
from torch.autograd.function import once_differentiable

from sft.distillation.loss import fp32_logits

from .targets import validate_targets


def projected_kl(logits, indices, probabilities):
    normalizer = logits.logsumexp(-1)
    selected = logits.gather(1, indices) - normalizer[:, None]
    other_logits = logits.scatter(1, indices, -torch.inf)
    other_normalizer = other_logits.logsumexp(-1)
    other_logp = other_normalizer - normalizer
    tail = (1 - probabilities.sum(-1)).clamp_min(0)
    selected_term = (
        torch.special.xlogy(probabilities, probabilities) - probabilities * selected
    )
    tail_term = torch.where(
        tail > 0, torch.special.xlogy(tail, tail) - tail * other_logp, 0
    )
    return selected_term.sum() + tail_term.sum(), other_logits, other_normalizer, tail


class SparseTeacherKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head, indices, probabilities, chunk):
        if head.requires_grad or type(chunk) is not int or chunk < 1:
            raise ValueError(
                "sparse teacher KL needs a frozen head and positive chunk size"
            )
        ctx.save_for_backward(hidden, head, indices, probabilities)
        ctx.chunk = chunk
        result = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(hidden), chunk):
            part, *_ = projected_kl(
                fp32_logits(hidden[start : start + chunk], head),
                indices[start : start + chunk],
                probabilities[start : start + chunk],
            )
            result += part
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        hidden, head, indices, probabilities = ctx.saved_tensors
        result = torch.empty_like(hidden)
        for start in range(0, len(hidden), ctx.chunk):
            stop = start + ctx.chunk
            logits = fp32_logits(hidden[start:stop], head)
            ix, q = indices[start:stop], probabilities[start:stop]
            _, other, log_other, tail = projected_kl(logits, ix, q)
            # Conditional student distribution over the residual bucket.
            # A fully enumerated vocabulary has an empty bucket and tail zero.
            conditional_other = (
                other - log_other.masked_fill(torch.isneginf(log_other), 0)[:, None]
            ).exp()
            ds = logits.softmax(-1) - tail[:, None] * conditional_other
            ds.scatter_add_(1, ix, -q)
            result[start:stop] = (ds * grad).to(hidden.dtype) @ head
        return result, None, None, None, None


def completion_teacher_losses(
    hidden, head, row, spans, chunk=64, tokenizer_sha256=None
):
    """One unweighted KL sum per completion, at state position target_position-1."""
    events = validate_targets(row, head.shape[0], tokenizer_sha256)
    result = []
    for prompt, tokens in spans:
        selected = [e for e in events if prompt <= e["position"] < prompt + tokens]
        if not selected:
            result.append(hidden[0, 0].sum() * 0)
            continue
        width = max(len(e["ids"]) for e in selected)
        # Pad by repeating a selected id with q=0; it never enlarges the bucket.
        ix = torch.tensor(
            [e["ids"] + [e["ids"][0]] * (width - len(e["ids"])) for e in selected],
            device=hidden.device,
            dtype=torch.long,
        )
        q = torch.tensor(
            [e["probabilities"] + [0.0] * (width - len(e["ids"])) for e in selected],
            device=hidden.device,
            dtype=torch.float32,
        )
        positions = torch.tensor(
            [e["position"] - 1 for e in selected], device=hidden.device
        )
        result.append(SparseTeacherKL.apply(hidden[0, positions], head, ix, q, chunk))
    return result
