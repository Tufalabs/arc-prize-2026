"""Full-vocabulary forward KL with bounded token-chunk memory."""

import math

import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

from training.flash.head_loss import cross_entropy_gradient


def fp32_logits(hidden, head):
    """Project BF16 activations without rounding logits before FP32 softmax."""
    if hidden.device.type == "cuda":
        return torch.mm(hidden, head.t(), out_dtype=torch.float32)
    # CPU tests and fallback runtimes do not implement CUDA's out_dtype path.
    return F.linear(hidden.float(), head.float())


class ChunkedDistillation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        student_head,
        teacher_hidden,
        teacher_head,
        targets,
        temperature,
        ce_weight,
        chunk,
    ):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(ce_weight) or ce_weight < 0 or chunk < 1:
            raise ValueError("invalid CE weight or chunk size")
        if (
            student_head.requires_grad
            or teacher_head.requires_grad
            or teacher_hidden.requires_grad
        ):
            raise ValueError("heads and teacher must be frozen")
        if (
            len(hidden) == 0
            or len(hidden) != len(teacher_hidden)
            or len(hidden) != len(targets)
        ):
            raise ValueError("completion alignment mismatch")
        if student_head.shape[0] != teacher_head.shape[0]:
            raise ValueError("vocabulary mismatch")
        ctx.save_for_backward(
            hidden, student_head, teacher_hidden, teacher_head, targets
        )
        ctx.temperature, ctx.ce_weight, ctx.chunk = temperature, ce_weight, chunk
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, len(hidden), chunk):
            stop = start + chunk
            s = fp32_logits(hidden[start:stop], student_head)
            t = fp32_logits(
                teacher_hidden[start:stop].to(hidden.device), teacher_head
            )
            log_t = (t / temperature).log_softmax(-1)
            log_s = (s / temperature).log_softmax(-1)
            total += (log_t.exp() * (log_t - log_s)).sum() * temperature**2
            if ce_weight:
                total += ce_weight * F.cross_entropy(
                    s, targets[start:stop], reduction="sum"
                )
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        hidden, sh, thidden, th, targets = ctx.saved_tensors
        out = torch.empty_like(hidden)
        for start in range(0, len(hidden), ctx.chunk):
            stop = min(start + ctx.chunk, len(hidden))
            s = fp32_logits(hidden[start:stop], sh)
            t = fp32_logits(thidden[start:stop].to(hidden.device), th)
            ds = ctx.temperature * (
                (s / ctx.temperature).softmax(-1) - (t / ctx.temperature).softmax(-1)
            )
            if ctx.ce_weight:
                ds += ctx.ce_weight * cross_entropy_gradient(s, targets[start:stop])
            out[start:stop] = (ds * grad).to(hidden.dtype) @ sh
        return out, None, None, None, None, None, None, None


def completion_distillation(
    hidden,
    student_head,
    teacher_hidden,
    teacher_head,
    ids,
    prompt_len,
    temperature=1.0,
    ce_weight=0.0,
    chunk=64,
):
    if not 1 <= prompt_len < ids.shape[1]:
        raise ValueError("empty prompt or completion")
    # State at p-1 predicts completion token p; include the final response token.
    return ChunkedDistillation.apply(
        hidden[0, prompt_len - 1 : -1].contiguous(),
        student_head,
        teacher_hidden,
        teacher_head,
        ids[0, prompt_len:],
        temperature,
        ce_weight,
        chunk,
    )


def objective_weights(policy, completion_tokens, global_tokens, global_examples, world):
    """Return loss-sum backward scale and sequence weight before rank averaging.

    Example mode averages per-sequence completion means; token mode averages all
    completion tokens. Use the actual example count, including partial GA tails.
    """
    if policy not in ("token", "example"):
        raise ValueError("loss_weighting must be token or example")
    if not 0 < completion_tokens <= global_tokens or global_examples < 1 or world < 1:
        raise ValueError("invalid objective counts")
    if policy == "token":
        return world / global_tokens, completion_tokens / global_tokens
    return world / (global_examples * completion_tokens), 1 / global_examples
