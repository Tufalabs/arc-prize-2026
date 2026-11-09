"""DPPO Binary-TV and sampled-token policy movement diagnostics.

The behavior logprobs are the captured rollout probabilities, never a
recomputed trainer anchor. See arXiv:2602.04879, equations 12, 13 and 23.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


def binary_tv_mask(
    logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("DPPO TV threshold must be finite and in (0, 1]")
    change = logprobs.detach().float().exp() - rollout_logprobs.detach().float().exp()
    return ~(
        ((advantages > 0) & (change > threshold))
        | ((advantages < 0) & (change < -threshold))
    )


def dppo_binary_tv_objective(
    logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    threshold: float,
    ratio_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 23's gradient with the paper's capped importance weight.

    The mask AND importance weight are detached. Clipping a differentiable
    ratio would incorrectly erase gradients beyond the cap. Full token mass
    stays in the loss denominator, including masked updates.
    """
    if not math.isfinite(ratio_cap) or ratio_cap < 1:
        raise ValueError("DPPO importance ratio cap must be finite and at least 1")
    mask = binary_tv_mask(logprobs, rollout_logprobs, advantages, threshold=threshold)
    log_ratio = logprobs.detach().float() - rollout_logprobs.detach().float()
    weight = log_ratio.clamp(max=math.log(ratio_cap)).exp()
    return mask * weight * advantages.detach() * logprobs, mask


@torch.no_grad()
def policy_update_statistics(
    logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    tv_threshold: float,
    ratio_cap: float,
) -> dict[str, float]:
    """Additive counts/sums plus maxima, reduced once across minibatches/ranks.

    These describe tokens actually used in forwards, including repeat epochs;
    they are not a full-distribution divergence or a final-policy evaluation.
    """
    if not logprobs.numel():
        return {}
    completion_mask = completion_mask.bool()
    old = rollout_logprobs.detach().float().exp()
    change = logprobs.detach().float().exp() - old
    log_ratio = logprobs.detach().float() - rollout_logprobs.detach().float()
    tv_blocked = ~binary_tv_mask(
        logprobs, rollout_logprobs, advantages, threshold=tv_threshold
    )
    capped = log_ratio > math.log(ratio_cap)
    groups = {"all": completion_mask}
    for sign, direction in (("positive", advantages > 0), ("negative", advantages < 0)):
        groups[sign] = completion_mask & direction
        for label, lower, upper in (
            ("p_lt_001", 0, 0.01),
            ("p_001_to_01", 0.01, 0.1),
            ("p_01_to_05", 0.1, 0.5),
            ("p_ge_05", 0.5, math.inf),
        ):
            groups[f"{sign}/{label}"] = (
                completion_mask & direction & (old >= lower) & (old < upper)
            )
    tensors = {}
    for name, mask in groups.items():
        tensors[f"{name}/tokens"] = mask.sum()
        tensors[f"{name}/abs_change_sum"] = change.abs().masked_fill(~mask, 0).sum()
        tensors[f"{name}/signed_change_sum"] = change.masked_fill(~mask, 0).sum()
        tensors[f"{name}/max_abs_change"] = change.abs().masked_fill(~mask, 0).max()
        tensors[f"{name}/log_ratio_sum"] = log_ratio.masked_fill(~mask, 0).sum()
        tensors[f"{name}/abs_log_ratio_sum"] = (
            log_ratio.abs().masked_fill(~mask, 0).sum()
        )
        tensors[f"{name}/max_abs_log_ratio"] = (
            log_ratio.abs().masked_fill(~mask, 0).max()
        )
        tensors[f"{name}/dppo_masked_count"] = (tv_blocked & mask).sum()
        tensors[f"{name}/ratio_capped_count"] = (capped & mask).sum()
    values = torch.stack([value.float() for value in tensors.values()]).cpu().tolist()
    return dict(zip(tensors, values, strict=True))


def merge_policy_update_statistics(
    accumulated: dict[str, float],
    incoming: Mapping[str, float],
) -> None:
    for name, value in incoming.items():
        if name.rsplit("/", 1)[-1].startswith("max_"):
            accumulated[name] = max(accumulated.get(name, 0.0), value)
        else:
            accumulated[name] = accumulated.get(name, 0.0) + value


def summarize_policy_update_statistics(stats: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for key, count in stats.items():
        if not key.endswith("/tokens"):
            continue
        prefix = key.removesuffix("tokens")
        result[key] = count
        if count == 0:
            continue  # An absent bucket has no measured error/fraction.
        if prefix == "score_centering/":
            for metric in (
                "relative_correction",
                "sampler_head_mass",
                "trainer_head_mass",
                "rho",
                "capped",
            ):
                result[prefix + "mean_" + metric] = (
                    stats[prefix + metric + "_sum"] / count
                )
            continue
        result[f"{prefix}max_abs_change"] = stats[f"{prefix}max_abs_change"]
        result[f"{prefix}max_abs_log_ratio"] = stats[f"{prefix}max_abs_log_ratio"]
        for metric, source in (
            ("mean_abs_change", "abs_change_sum"),
            ("mean_signed_change", "signed_change_sum"),
            ("mean_log_ratio", "log_ratio_sum"),
            ("mean_abs_log_ratio", "abs_log_ratio_sum"),
            ("dppo_masked_fraction", "dppo_masked_count"),
            ("ratio_capped_fraction", "ratio_capped_count"),
        ):
            result[prefix + metric] = stats[prefix + source] / count
    return result
