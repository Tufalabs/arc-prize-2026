"""The fast QSA indexer in a chunked prefill selects what the reference selects."""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
configuration = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")

from training.flash_qsa_train import _fast_forward  # noqa: E402


class StubCache:
    """The one cache method the indexer uses: append this chunk's raw keys."""

    def __init__(self):
        self.keys = {}

    def update_indexer(self, raw_keys, layer_idx):
        previous = self.keys.get(layer_idx)
        self.keys[layer_idx] = raw_keys if previous is None else torch.cat([previous, raw_keys], dim=1)
        return self.keys[layer_idx]


def indexer():
    config = configuration.Qwen4ExpTextConfig(
        hidden_size=16,
        head_dim=32,
        partial_rotary_factor=0.25,  # rotary dim 8 = the index head
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
    )
    torch.manual_seed(0)
    module = modeling.Qwen4ExpTextQSAIndexer(config, layer_idx=0)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter)
    return module


def rotary(length):
    angles = torch.arange(length)[:, None] * torch.linspace(0.1, 1.0, 4)[None, :]
    angles = torch.cat([angles, angles], dim=-1)[None]
    return angles.cos(), angles.sin()


def causal(queries, keys):
    return torch.ones(queries, keys, dtype=torch.bool).tril_(keys - queries)[None, None]


@pytest.mark.parametrize("split", [8, 24, 37])
def test_chunked_fast_selection_equals_the_one_shot_reference(split):
    module = indexer()
    length = 40
    hidden = torch.randn(1, length, 16)
    cos, sin = rotary(length)
    with torch.no_grad():
        reference = module(hidden, (cos, sin), causal(length, length), None)
        cache = StubCache()
        first = _fast_forward(module, hidden[:, :split], (cos[:, :split], sin[:, :split]), causal(split, split), cache)
        second = _fast_forward(module, hidden[:, split:], (cos, sin), causal(length - split, length), cache)
    assert torch.equal(first[..., :split], reference[:, :, :split, :split])
    assert torch.equal(second, reference[:, :, split:, :])


def test_the_reference_agrees_with_itself_under_a_cache():
    """Sanity: the reference's own chunked path is the target the fast one meets."""
    module = indexer()
    hidden = torch.randn(1, 32, 16)
    cos, sin = rotary(32)
    with torch.no_grad():
        whole = module(hidden, (cos, sin), causal(32, 32), None)
        cache = StubCache()
        module(hidden[:, :20], (cos[:, :20], sin[:, :20]), causal(20, 20), cache)
        tail = module(hidden[:, 20:], (cos, sin), causal(12, 32), cache)
    assert torch.equal(tail, whole[:, :, 20:, :])


def test_a_mask_that_is_not_causal_at_the_offset_is_refused():
    module = indexer()
    hidden = torch.randn(1, 8, 16)
    cos, sin = rotary(16)
    cache = StubCache()
    cache.update_indexer(torch.randn(1, 8, 8), 0)
    with pytest.raises(RuntimeError, match="unpadded causal mask"):
        _fast_forward(module, hidden, (cos, sin), torch.ones(1, 1, 8, 16, dtype=torch.bool), cache)
