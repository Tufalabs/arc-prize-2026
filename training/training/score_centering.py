"""Detached score centering and the compact sampler-head evidence contract."""

import base64
import math

import numpy as np


def validate_head(ids, logprobs, *, tokens, k, vocab_size):
    if (
        ids.shape != (tokens, k)
        or logprobs.shape != (tokens, k)
        or ids.dtype != np.dtype("<i4")
        or logprobs.dtype != np.dtype("<f4")
        or np.any(ids < 0)
        or np.any(ids >= vocab_size)
        or not np.isfinite(logprobs).all()
        or np.any(logprobs > 0)
    ):
        raise ValueError(
            "invalid sampler top-k shape, dtype, vocabulary IDs or logprobs"
        )
    if k > 1 and np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
        raise ValueError("sampler top-k IDs must be unique per token")


def pack_head(rows, *, k, vocab_size):
    ids = np.empty((len(rows), k), dtype="<i4")
    logprobs = np.empty((len(rows), k), dtype="<f4")
    for index, row in enumerate(rows):
        # vLLM includes the sampled token even when it is outside the top-k.
        head = sorted(row.items(), key=lambda item: item[1].logprob, reverse=True)[:k]
        if len(head) != k:
            raise ValueError("engine omitted sampler top-k entries")
        for column, (token, value) in enumerate(head):
            if type(token) is not int or not 0 <= token < vocab_size:
                raise ValueError("invalid sampler top-k token ID")
            if not math.isfinite(value.logprob) or value.logprob > 0:
                raise ValueError("invalid sampler top-k logprob")
            ids[index, column] = token
            logprobs[index, column] = value.logprob
    validate_head(ids, logprobs, tokens=len(rows), k=k, vocab_size=vocab_size)
    return {
        "k": k,
        "ids_i32le": base64.b64encode(ids.tobytes()).decode("ascii"),
        "logprobs_f32le": base64.b64encode(logprobs.tobytes()).decode("ascii"),
    }


def unpack_head(payload, *, tokens, k, vocab_size):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"k", "ids_i32le", "logprobs_f32le"}
        or type(payload["k"]) is not int
        or payload["k"] != k
    ):
        raise ValueError("sampler top-k wire schema or k differs from policy")
    arrays = []
    for key, dtype in (("ids_i32le", "<i4"), ("logprobs_f32le", "<f4")):
        encoded = payload[key]
        if not isinstance(encoded, str) or len(encoded) != 4 * (
            (tokens * k * 4 + 2) // 3
        ):
            raise ValueError("sampler top-k byte length differs from completion")
        raw = base64.b64decode(encoded, validate=True)
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(tokens, k))
    validate_head(*arrays, tokens=tokens, k=k, vocab_size=vocab_size)
    return tuple(arrays)


def score_centered_objective(
    logp, old_logp, head_logp, sampler_head_logp, advantage, *, cap=2.0, eps=1e-6
):
    """Eq. 12 with TIS; cap=inf also permits checking the exact-IS identity."""
    import torch

    # Clamping before exp implements TIS without overflowing off-policy ratios.
    with torch.no_grad():
        weights = (head_logp - sampler_head_logp).clamp(max=math.log(cap)).exp()
        sampled_weight = (logp - old_logp).clamp(max=math.log(cap)).exp()
        p, q = head_logp.exp(), sampler_head_logp.exp()
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        rho = (1 - q_mass).clamp(min=eps) / (1 - p_mass).clamp(min=eps)
        alpha = rho * (1 / rho).clamp(max=cap)
        coefficients = q * weights - alpha.unsqueeze(-1) * p
    correction = (coefficients * head_logp).sum(-1)
    sampled_term = sampled_weight * logp
    return advantage * (sampled_term - correction), {
        "relative_correction": correction.detach().abs()
        / sampled_term.detach().abs().clamp(min=eps),
        "sampler_head_mass": q_mass,
        "trainer_head_mass": p_mass,
        "rho": rho,
    }
