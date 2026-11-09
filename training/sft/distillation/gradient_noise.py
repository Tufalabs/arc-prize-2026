"""Global-only objective-weighted gradient signal/noise proxy before clipping/AdamW.

Assumes independent sequence-mean gradients with a common mean and covariance,
conditional on completion lengths. Correlated traces and length-dependent gradient
statistics violate this approximation; this is not a token-iid variance estimate.
"""

import copy
import math

import torch

SCHEMA = {
    "version": 2,
    "ema_alpha": 0.05,
    "estimator": "global_weighted_sequence_moments",
}


@torch.no_grad()
def local_moments(routers, local_tokens, global_tokens, world, *, sequence_weight=None):
    """Return weighted squared norms plus sum-of-squared-weights contribution.

    Trainer's local gradient is world * weight * g_i, with token-length
    weights by default or an explicit objective weight for equal-example loss.
    Here g_i is the sequence's mean completion gradient. Undo that scale only
    in this diagnostic; never modify the actual gradients.
    """
    if world < 1 or not 0 < local_tokens < global_tokens:
        raise ValueError("noise estimation requires multiple nonempty sequences")
    weight = (
        local_tokens / global_tokens if sequence_weight is None else sequence_weight
    )
    if not math.isfinite(weight) or not 0 < weight < 1:
        raise ValueError("invalid sequence weight")
    norms = []
    for name, parameter in routers.items():
        if parameter.grad is None:
            raise ValueError("missing router gradient: " + name)
        norm2 = torch.linalg.vector_norm(parameter.grad, dtype=torch.float64).square()
        norms.append(norm2 / (world * world * weight))
    if not norms:
        raise ValueError("no routers")
    return torch.stack([torch.stack(norms).sum(), norms[0].new_tensor(weight * weight)])


def components(second_moment, mean_norm2, weight_square_sum):
    """Weighted unbiased covariance trace under the documented assumptions.

    A=sum(w_i ||g_i||²), M=||sum(w_i g_i)||², q=sum(w_i²).
    Covariance trace V=(A-M)/(1-q); batch-mean noise N=q*V;
    bias-corrected signal S=M-N. S may be negative with few observations.
    """
    if not all(
        math.isfinite(x) for x in (second_moment, mean_norm2, weight_square_sum)
    ):
        raise ValueError("nonfinite gradient moment")
    if second_moment < 0 or mean_norm2 < 0 or not 0 < weight_square_sum < 1:
        raise ValueError("invalid gradient moment or rank weights")
    # Actual all-reduce averages FP32 gradients; allow its small rounding error.
    residual = second_moment - mean_norm2
    if residual < -1e-5 * max(second_moment, mean_norm2, 1e-300):
        raise ValueError("inconsistent gradient moments")
    noise = weight_square_sum * max(0.0, residual) / (1 - weight_square_sum)
    return {
        "mean_gradient_norm_squared": mean_norm2,
        "noise_power": noise,
        "signal_power": mean_norm2 - noise,
    }


def ratio_metrics(values):
    # Do not turn a zero/unresolved signal into a huge epsilon-dependent number.
    signal, noise = values["signal_power"], values["noise_power"]
    floor = 1e-12 * max(values["mean_gradient_norm_squared"], noise)
    resolved = signal > floor
    return {
        **values,
        "signal_resolved": resolved,
        "signal_to_noise": signal / noise if resolved and noise > floor else None,
        "noise_resolved": noise > floor,
    }


class GradientNoise:
    def __init__(self, router_names):
        self.names = list(router_names)
        self.state = {
            "schema": dict(SCHEMA),
            "router_names": self.names,
            "observations": 0,
            "ema": {},
        }

    def restore(self, state):
        if state["schema"] != SCHEMA or state["router_names"] != self.names:
            raise ValueError("gradient-noise checkpoint schema/router mismatch")
        self.state = copy.deepcopy(state)

    @torch.no_grad()
    def finish(self, routers, reduced_moments):
        """Call after router gradient averaging, before clipping; all ranks agree."""
        if list(routers) != self.names or len(reduced_moments) != 2:
            raise ValueError("gradient-noise router ordering changed")
        second_moment, q = reduced_moments.detach().cpu().tolist()
        mean_norm2 = (
            torch.stack(
                [
                    torch.linalg.vector_norm(p.grad, dtype=torch.float64).square()
                    for p in routers.values()
                ]
            )
            .sum()
            .item()
        )
        current = {"global": components(second_moment, mean_norm2, q)}
        result = {
            "effective_sequences": 1 / q,
            "observations": self.state["observations"] + 1,
            "scopes": {},
        }
        for scope, values in current.items():
            old = self.state["ema"].get(scope)
            smooth = {
                k: (SCHEMA["ema_alpha"] * v + (1 - SCHEMA["ema_alpha"]) * old[k])
                if old is not None
                else v
                for k, v in values.items()
            }
            self.state["ema"][scope] = smooth
            result["scopes"][scope] = {
                "raw": ratio_metrics(values),
                "ema": ratio_metrics(smooth),
            }
        self.state["observations"] += 1
        return result


def wandb_metrics(report):
    out = {
        "train/gradient_noise/effective_sequences": report["effective_sequences"],
        "train/gradient_noise/observations": report["observations"],
    }
    for scope, estimates in report["scopes"].items():
        for kind, values in estimates.items():
            for key, value in values.items():
                if value is not None:
                    out[f"train/gradient_noise/{scope}/{kind}/{key}"] = value
    return out
