"""Exact restoration checks and bounded-memory numerical comparisons."""

import hashlib
from pathlib import Path

import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def assert_tree_exact(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert torch.equal(actual.detach().cpu(), expected.cpu()), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            assert_tree_exact(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_tree_exact(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, (path, actual, expected)


def tree_digest(value):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            raw = x.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            h.update(raw.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())

    visit(value)
    return h.hexdigest()


def tensor_diff(actual, expected):
    a, e = actual.detach().cpu(), expected.detach().cpu()
    assert a.shape == e.shape and a.dtype == e.dtype
    a, e = a.reshape(-1), e.reshape(-1)
    changed, ss, refss, maxabs = 0, 0.0, 0.0, 0.0
    for start in range(0, a.numel(), 1048576):
        aa, ee = (
            a[start : start + 1048576].double(),
            e[start : start + 1048576].double(),
        )
        delta = aa - ee
        assert torch.isfinite(delta).all()
        changed += int((aa != ee).sum())
        ss += delta.square().sum().item()
        refss += ee.square().sum().item()
        if delta.numel():
            maxabs = max(maxabs, delta.abs().max().item())
    return {
        "exact": changed == 0,
        "changed": changed,
        "elements": a.numel(),
        "max_abs": maxabs,
        "relative_l2": (ss / max(refss, 1e-300)) ** 0.5,
    }


def compare_tensors(actual, expected):
    assert actual.keys() == expected.keys()
    detail = {k: tensor_diff(actual[k], expected[k]) for k in actual}
    return {
        "exact": all(d["exact"] for d in detail.values()),
        "max_abs": max((d["max_abs"] for d in detail.values()), default=0),
        "per_tensor": detail,
    }


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu_tree(value), tmp)
    tmp.replace(path)


def gradient_state(routers):
    return {k: p.grad.detach().cpu().clone() for k, p in routers.items()}


def optimizer_tensor_state(optimizer, names):
    state = optimizer.state_dict()
    ids = [i for group in state["param_groups"] for i in group["params"]]
    assert len(ids) == len(names)
    return {
        f"{name}.{field}": value
        for name, i in zip(names, ids)
        for field, value in state["state"][i].items()
        if isinstance(value, torch.Tensor)
    }
