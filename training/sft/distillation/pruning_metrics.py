"""Deterministic, tie-aware pruning diagnostics. Thresholds are advisory only."""

import math


def bottom(scores, k):
    if not 0 < k < len(scores) or any(not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError("invalid candidate budget or expert scores")
    return sorted(range(len(scores)), key=lambda i: (scores[i], i))[:k]


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def spearman(a, b):
    if len(a) != len(b):
        raise ValueError("rank lengths differ")
    a, b = ranks(a), ranks(b)
    mean = (len(a) - 1) / 2
    a, b = [v - mean for v in a], [v - mean for v in b]
    den = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / den if den else None


def layer_metrics(scores, previous, history, k):
    candidates = bottom(scores, k)
    sets = [set(bottom(s, k)) for s in history] + [set(candidates)]
    persistence = [sum(i in s for s in sets) / len(sets) for i in range(len(scores))]
    ordered = sorted(scores)
    gap = ordered[k] - ordered[k - 1]
    total = sum(scores)
    return {
        "candidates": candidates,
        "overlap_previous": len(set(candidates) & set(bottom(previous, k))) / k
        if previous is not None
        else None,
        "overlap_window_start": len(set(candidates) & sets[0]) / k
        if len(sets) > 1
        else None,
        "spearman": spearman(scores, previous) if previous is not None else None,
        "persistence": persistence,
        "candidate_persistence_min": min(persistence[i] for i in candidates),
        "candidate_persistence_mean": sum(persistence[i] for i in candidates) / k,
        "boundary_gap": gap,
        "boundary_gap_normalized": gap / total if total else 0.0,
        "boundary_tied": gap == 0,
        "window_evaluations": len(sets),
    }


def route_overlap(current, previous):
    """Per-token intersection/top-k, independent of top-k ordering."""
    if current.shape != previous.shape:
        raise ValueError("fixed routing sample shape changed")
    return (current.unsqueeze(-1) == previous.unsqueeze(-2)).any(-1).float().mean(-1)
