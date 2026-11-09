"""Actual global router update/weight L2 ratio; computed on one replica only."""

import math

import torch


@torch.no_grad()
def snapshot_routers(routers):
    return {name: p.detach().clone() for name, p in routers.items()}


@torch.no_grad()
def router_update_ratio(routers, before):
    """Consume the snapshot as scratch; never modify parameters or gradients.

    All router tensors form one vector. The denominator is the pre-update
    weight norm. A zero denominator is undefined, represented as None.
    """
    if not routers or list(routers) != list(before):
        raise ValueError("router snapshot census mismatch")
    weight_squares, update_squares = [], []
    for name, p in routers.items():
        old = before[name]
        if old.shape != p.shape or old.dtype != p.dtype or old.device != p.device:
            raise ValueError("router snapshot shape/dtype/device mismatch")
        weight_squares.append(
            torch.linalg.vector_norm(old, dtype=torch.float64).square()
        )
        old.sub_(p.detach())
        update_squares.append(
            torch.linalg.vector_norm(old, dtype=torch.float64).square()
        )
    weights = torch.stack(weight_squares).sum()
    updates = torch.stack(update_squares).sum()
    values = torch.stack([weights, updates]).cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite router update ratio")
    return (values[1] / values[0]) ** 0.5 if values[0] > 0 else None
