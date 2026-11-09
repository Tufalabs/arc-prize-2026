"""Masking pruned experts routes exactly as the physically pruned router does."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from training.flash_route_stats import RouterMask  # noqa: E402


def router(experts, *, norm):
    config = SimpleNamespace(num_experts_per_tok=2, num_experts=experts, norm_topk_prob=norm, hidden_size=4)
    return modeling.Qwen4ExpTextTopKRouter(config)


@pytest.mark.parametrize("norm", [True, False])
def test_masked_routing_equals_the_pruned_router(norm):
    torch.manual_seed(3)
    full = router(8, norm=norm)
    torch.nn.init.normal_(full.weight)
    kept = [1, 3, 4, 6]
    pruned = router(4, norm=norm)
    with torch.no_grad():
        pruned.weight.copy_(full.weight[kept])
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(gate=full))]))
    mask = RouterMask(model)
    hidden = torch.randn(5, 4)
    unmasked = full(hidden)
    mask.set({"kept": {"0": kept}})
    _, scores, indices = full(hidden)
    _, pruned_scores, pruned_indices = pruned(hidden)
    assert torch.equal(indices, torch.tensor(kept)[pruned_indices])
    assert torch.allclose(scores, pruned_scores, atol=1e-6)
    mask.set(None)
    assert torch.equal(full(hidden)[2], unmasked[2])
