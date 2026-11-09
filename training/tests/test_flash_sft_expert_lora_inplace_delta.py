"""Flash SFT expert_lora_inplace_delta: routed-expert LoRA deltas added in place give the same values and gradients."""

import pytest
import torch
from torch import nn

from training.flash.expert_lora import (
    add_lora_delta, enable_inplace_delta, install_expert_lora, validate_expert_lora_inplace_delta,
)

LORA = {"rank": 3, "alpha": 6.0, "dropout": 0.0}


def experts(dtype):
    torch.manual_seed(7)
    m = nn.Module()
    m.gate_up_proj = nn.Parameter(torch.randn(3, 32, 16, dtype=dtype), requires_grad=False)
    m.down_proj = nn.Parameter(torch.randn(3, 16, 16, dtype=dtype), requires_grad=False)
    m.num_experts = 3
    m._apply_gate = lambda z: torch.nn.functional.silu(z.chunk(2, -1)[0]) * z.chunk(2, -1)[1]
    install_expert_lora(m, LORA)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("_B"):
                p.normal_(std=0.05)
    return m


def grouped(inputs, w, offsets):
    starts = [0] + offsets.tolist()
    return torch.cat([inputs[starts[i]:starts[i + 1]] @ w[i].T for i in range(len(w))])


def run(m, dtype, optimization):
    from training.flash.pointwise_optimizations import expert_forward_with_weights

    g = torch.Generator().manual_seed(3)
    x = torch.randn(9, 16, generator=g).to(dtype).requires_grad_()
    ids = torch.tensor([[0, 2], [1, 2], [0, 1]] * 3)
    w = torch.rand(9, 2, generator=g).to(dtype).requires_grad_()
    out = expert_forward_with_weights(m, x, ids, w, m.gate_up_proj, m.down_proj, optimization=optimization,
                                      grouped_linear=grouped)
    upstream = torch.randn(out.shape, generator=g).to(dtype)
    params = [x, w] + [p for p in m.parameters() if p.requires_grad]
    return out, torch.autograd.grad(out, params, upstream)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("optimization", ["weighted_permutation", "no_sentinel"])
def test_values_and_gradients_identical_to_out_of_place(dtype, optimization):
    m = experts(dtype)
    out, grads = run(m, dtype, optimization)
    assert enable_inplace_delta(m) == 1
    out_inplace, grads_inplace = run(m, dtype, optimization)
    assert torch.equal(out, out_inplace)
    assert len(grads) == len(grads_inplace) == 8
    for a, b in zip(grads, grads_inplace):
        assert torch.equal(a, b)


def test_column_add_passes_the_gradient_through_without_copyslices():
    m = experts(torch.float32)
    x = torch.randn(4, 16, requires_grad=True)
    offsets = torch.tensor([1, 3, 4], dtype=torch.int32)
    up = grouped(x, m.gate_up_proj, offsets)
    up = add_lora_delta(up, m, "gate", x, offsets, grouped, (0, 16))
    up = add_lora_delta(up, m, "up", x, offsets, grouped, (16, 32))
    names = []
    node = up.grad_fn
    while node is not None and len(names) < 12:
        names.append(type(node).__name__)
        node = node.next_functions[0][0] if node.next_functions else None
    assert not any("CopySlices" in n for n in names), names
    assert names[0] == "_AddIntoColumnsBackward"


def test_validation_and_enable_without_adapters():
    validate_expert_lora_inplace_delta(False, None)
    validate_expert_lora_inplace_delta(True, LORA)
    for args in ((True, None), ("yes", LORA)):
        with pytest.raises(ValueError):
            validate_expert_lora_inplace_delta(*args)
    with pytest.raises(ValueError):
        enable_inplace_delta(nn.Linear(2, 2))
