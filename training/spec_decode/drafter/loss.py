"""Full-vocabulary distillation against the target, in bounded token chunks,
and the acceptance metrics read off the same logits.

``ChunkedDistillation`` is our (arc-training ``sft/distillation/loss.py``
on ``flash-router-distillation-production``): forward
KL(teacher || student) at a temperature, optional CE, recomputing each chunk's
logits in the backward instead of keeping them. Here the student's hidden is
the head's single stream, and teacher and student share the target's LM head.
One addition: optional per-row ``weights`` (default none, which is upstream's
sum), for a loss weighted by how much each drafted position counts in serving.
"""

import math

import torch
import torch.nn.functional as F


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
        weights=None,
    ):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(ce_weight) or ce_weight < 0 or chunk < 1:
            raise ValueError("invalid CE weight or chunk size")
        if student_head.requires_grad or teacher_head.requires_grad or teacher_hidden.requires_grad:
            raise ValueError("heads and teacher must be frozen")
        if len(hidden) == 0 or len(hidden) != len(teacher_hidden) or len(hidden) != len(targets):
            raise ValueError("completion alignment mismatch")
        if student_head.shape[0] != teacher_head.shape[0]:
            raise ValueError("vocabulary mismatch")
        if weights is not None and (weights.shape != (len(hidden),) or weights.requires_grad):
            raise ValueError("weights must be one frozen value a row")
        if weights is None:
            weights = torch.ones(len(hidden), dtype=torch.float32, device=hidden.device)
        ctx.save_for_backward(hidden, student_head, teacher_hidden, teacher_head, targets, weights.float())
        ctx.temperature, ctx.ce_weight, ctx.chunk = temperature, ce_weight, chunk
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, len(hidden), chunk):
            stop = start + chunk
            s = F.linear(hidden[start:stop], student_head).float()
            t = F.linear(teacher_hidden[start:stop].to(hidden.device), teacher_head).float()
            log_t = (t / temperature).log_softmax(-1)
            log_s = (s / temperature).log_softmax(-1)
            w = weights[start:stop].float()
            total += ((log_t.exp() * (log_t - log_s)).sum(-1) * w).sum() * temperature**2
            if ce_weight:
                total += ce_weight * (F.cross_entropy(s, targets[start:stop], reduction="none") * w).sum()
        return total

    @staticmethod
    def backward(ctx, grad):
        hidden, sh, thidden, th, targets, weights = ctx.saved_tensors
        out = torch.empty_like(hidden)
        for start in range(0, len(hidden), ctx.chunk):
            stop = min(start + ctx.chunk, len(hidden))
            s = F.linear(hidden[start:stop], sh).float()
            t = F.linear(thidden[start:stop].to(hidden.device), th).float()
            ds = ctx.temperature * ((s / ctx.temperature).softmax(-1) - (t / ctx.temperature).softmax(-1))
            if ctx.ce_weight:
                ce = s.softmax(-1)
                ce[torch.arange(stop - start, device=s.device), targets[start:stop]] -= 1
                ds += ctx.ce_weight * ce
            ds *= weights[start:stop, None]
            out[start:stop] = (ds * grad).to(hidden.dtype) @ sh
        return out, None, None, None, None, None, None, None, None


#: The recordings' sampling, both harnesses (``sampling_params`` in the traces).
SERVED_SAMPLING = {"temperature": 0.6, "top_k": 194, "top_p": 0.8899009607194754, "presence_penalty": 1.5}


def _top_k_top_p(logits, top_k, top_p):
    """vLLM's masking (``apply_top_k_top_p_pytorch``): keep the ``top_k``
    largest, then the smallest top set whose probability reaches ``top_p``;
    the largest always survives."""
    ordered, index = logits.sort(dim=-1, descending=False)
    vocab = logits.shape[-1]
    kth = ordered[:, vocab - top_k].unsqueeze(-1)
    ordered = ordered.masked_fill(ordered < kth, -float("inf"))
    cumulative = ordered.softmax(-1).cumsum(-1)
    drop = cumulative <= 1 - top_p
    drop[:, -1] = False
    ordered = ordered.masked_fill(drop, -float("inf"))
    return torch.empty_like(logits).scatter_(-1, index, ordered)


@torch.no_grad()
def acceptance_metrics(
    hidden, teacher_hidden, head, targets, temperatures=(1.0, 0.6), chunk=256, served=None, ratios=None
):
    """Sums over rows, for averaging across chains: KL at T=1, top-1
    agreement, how often the student's argmax is the token the target
    sampled, and expected acceptance ``sum_v min(p, q)`` on raw logits at each
    temperature.

    With ``served`` -- ``(ids, reply_start, label_position)``, the chain's ids
    and per row where its call's reply starts and which position it drafts --
    also ``accept_served``: p as the V2 sampler processes the target's logits
    before verifying (presence penalty over the reply so far, temperature,
    top-k, top-p; ``SERVED_SAMPLING``), q as the draft is sampled
    (temperature only).

    With ``ratios`` (a dict), also each row's acceptance ratio for the token
    the data holds, ``r = min(1, q(x) / p(x))``: under ``"t0.6"`` with p at
    T=0.6 alone, under ``"served"`` (with ``served``) with p as the V2
    sampler processes it. Where the data's tokens were sampled from that p,
    the mean over a draft's steps of the running product of ``r`` is exactly
    the probability that the served verifier accepts that far (the draft is
    proposed and kept with probability ``min(p, q)`` per token, which is
    ``p(x) r(x)``): the per-draft acceptance without the offline average's
    assumptions."""
    if ratios is not None:
        ratios.setdefault("t0.6", [])
        if served is not None:
            ratios.setdefault("served", [])
    sums = {"rows": 0, "kl": 0.0, "top1": 0.0, "argmax_is_sample": 0.0}
    for temperature in temperatures:
        sums[f"accept_t{temperature:g}"] = 0.0
    if served is not None:
        sums["accept_served"] = 0.0
        sums["ceiling_served"] = 0.0
    for start in range(0, len(hidden), chunk):
        stop = start + chunk
        s = F.linear(hidden[start:stop], head).float()
        t = F.linear(teacher_hidden[start:stop], head).float()
        log_t, log_s = t.log_softmax(-1), s.log_softmax(-1)
        sums["kl"] += float((log_t.exp() * (log_t - log_s)).sum())
        top_s = s.argmax(-1)
        sums["top1"] += float((top_s == t.argmax(-1)).sum())
        sums["argmax_is_sample"] += float((top_s == targets[start:stop]).sum())
        for temperature in temperatures:
            p = (t / temperature).softmax(-1)
            q = (s / temperature).softmax(-1)
            sums[f"accept_t{temperature:g}"] += float(torch.minimum(p, q).sum())
        if ratios is not None:
            rows, x = torch.arange(len(s), device=s.device), targets[start:stop]
            log_q = (s / SERVED_SAMPLING["temperature"]).log_softmax(-1)[rows, x]
            log_p = (t / SERVED_SAMPLING["temperature"]).log_softmax(-1)[rows, x]
            ratios["t0.6"].append((log_q - log_p).clamp_max(0).exp())
        if served is not None:
            ids, reply_start, label_position = served
            first, last = reply_start[start:stop], label_position[start:stop]
            width = int((last - first).max())
            offsets = torch.arange(width, device=ids.device)
            window = first[:, None] + offsets[None]
            present = offsets[None] < (last - first)[:, None]
            tokens = ids[window.clamp_max(len(ids) - 1)]
            seen = torch.zeros_like(t, dtype=torch.int32).scatter_add_(1, tokens, present.to(torch.int32)) > 0
            config = SERVED_SAMPLING
            processed = (t - config["presence_penalty"] * seen) / config["temperature"]
            p = _top_k_top_p(processed, config["top_k"], config["top_p"]).softmax(-1)
            q = (s / config["temperature"]).softmax(-1)
            sums["accept_served"] += float(torch.minimum(p, q).sum())
            if ratios is not None:
                # A token outside the served top set has p = 0 there: text
                # the served sampler did not write, where the ratio means nothing.
                ratios["served"].append((log_q - p[rows, x].log()).clamp_max(0).exp())
            # The target itself as drafter: served drafts are sampled at the
            # temperature alone, so even q = the target's own tempered
            # distribution loses where the verifier's processing moves p.
            own = (t / config["temperature"]).softmax(-1)
            sums["ceiling_served"] += float(torch.minimum(p, own).sum())
        sums["rows"] += len(s)
    if ratios is not None:
        for name in ratios:
            ratios[name] = torch.cat(ratios[name]) if ratios[name] else torch.zeros(0, device=hidden.device)
    return sums
