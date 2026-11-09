"""Common FP32-oracle RMS/max gate and bitwise repeatability checks."""

import json

import torch


def reference_norm(x, weight, groups, eps=1e-6):
    xf = x.float().unflatten(-1, (groups, x.shape[-1] // groups))
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y.flatten(-2) * (1 + weight.float())).to(x.dtype)


def reference_mix(x, up, groups):
    return (x * up.sigmoid()).unflatten(-1, (groups, x.shape[-1] // groups)).mean(-2)


def reference_inject(x, h, w):
    return x + (h.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)


def error(actual, reference):
    return (actual.float() - reference).square().mean().sqrt().item()


def same_bits(actual, expected):
    return (actual.dtype == expected.dtype and actual.shape == expected.shape
            and torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                            expected.contiguous().reshape(-1).view(torch.uint8)))


def check_packed(test, operation, inputs, lengths=(1, 73, 183)):
    """Token-local fused outputs and adjoints must respect document boundaries."""
    inputs = [value.detach().requires_grad_() for value in inputs]
    whole = operation(*inputs)
    whole = whole if isinstance(whole, tuple) else (whole,)
    pieces = [operation(*values) for values in zip(*(x.split(lengths, 1) for x in inputs))]
    pieces = [values if isinstance(values, tuple) else (values,) for values in pieces]
    packed = tuple(torch.cat(values, 1) for values in zip(*pieces))
    upstream = tuple(torch.randn_like(value) for value in whole)
    full_grad = torch.autograd.grad(whole, inputs, upstream)
    packed_grad = torch.autograd.grad(packed, inputs, upstream)
    for a, b in zip((*whole, *full_grad), (*packed, *packed_grad), strict=True):
        test.assertTrue(same_bits(a, b), "Packed output/adjoint differs from independent documents")


def check_op(self, eager, fused, inputs, *, name=None):
    native = [x.detach().requires_grad_() for x in inputs]
    high = [x.detach().float().requires_grad_() for x in inputs]
    output = eager(*native)
    oracle = eager(*high)
    output = output if isinstance(output, tuple) else (output,)
    oracle = oracle if isinstance(oracle, tuple) else (oracle,)
    upstream = tuple(torch.randn_like(value) for value in output)
    expected = torch.autograd.grad(oracle, high, tuple(value.float() for value in upstream))
    baseline = torch.autograd.grad(output, native, upstream)
    previous = None
    records = []
    for _ in range(2):
        actual = fused(*native)
        actual = actual if isinstance(actual, tuple) else (actual,)
        gradients = torch.autograd.grad(actual, native, upstream)
        for value_name, a, b, ref in zip(
            [f"output{i}" for i in range(len(output))] + [f"gradient{i}" for i in range(len(inputs))],
            (*actual, *gradients), (*output, *baseline), (*oracle, *expected),
        ):
            eb, ea = error(b, ref), error(a, ref)
            mb = (b.float() - ref).abs().max().item()
            ma = (a.float() - ref).abs().max().item()
            if previous is None:
                # Callable instances and partials have no __name__.
                record = {"op": name or getattr(fused, "__name__", type(fused).__name__),
                          "shape": list(inputs[0].shape), "value": value_name,
                          "eager_rms": eb, "fused_rms": ea,
                          "eager_max": mb, "fused_max": ma}
                records.append(record)
                print(json.dumps(record), flush=True)
            self.assertLessEqual(ea, 1.5 * eb,
                                 f"{value_name}: candidate={ea}, eager={eb}")
            self.assertLessEqual(ma, 1.5 * mb,
                                 f"{value_name} max: candidate={ma}, eager={mb}")
        current = tuple(value.detach() for value in (*actual, *gradients))
        if previous is not None:
            self.assertTrue(all(same_bits(a, b) for a, b in zip(current, previous)),
                            "Forward/backward must be bitwise deterministic")
        previous = current
    return records
