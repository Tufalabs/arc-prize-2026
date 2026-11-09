from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from training.flash.dense_lora import (
    LoRALinear, adapters_disabled, dense_lora_policy,
    to_peft_state_dict, from_peft_state_dict,
)


def test_targets_and_disabled_context():
    layer = torch.nn.Module()
    layer.linear_attn = torch.nn.Module()
    for name in ("in_proj_qkv", "in_proj_a", "in_proj_b"):
        setattr(layer.linear_attn, name, torch.nn.Linear(4, 4))
    policy = dense_lora_policy(r=2, target_modules=["in_proj_a", "in_proj_b"])
    params = policy.configure_layer(layer, 0)
    assert len(params) == 4
    assert not isinstance(layer.linear_attn.in_proj_qkv, LoRALinear)
    module = layer.linear_attn.in_proj_a
    module.lora_B.data.fill_(1)
    x = torch.randn(3, 4)
    original = module(x)
    with pytest.raises(RuntimeError), adapters_disabled(layer):
        with adapters_disabled(layer):
            assert torch.equal(module(x), module.base(x))
        assert torch.equal(module(x), module.base(x))
        raise RuntimeError("restore on exception")
    assert torch.equal(module(x), original)


def test_peft_roundtrip_and_runtime_export(tmp_path):
    import json
    from safetensors.torch import save_file
    from training.runtime_lora import export_runtime_lora_adapter

    config = SimpleNamespace(r=2, target_modules=["q_proj", "in_proj_qkv", "in_proj_b", "gate_proj"])
    params = {}
    for parent, leaf in [("self_attn", "q_proj"), ("linear_attn", "in_proj_qkv"),
                         ("linear_attn", "in_proj_b"), ("mlp.shared_expert", "gate_proj")]:
        for side, shape in [("A", (2, 4)), ("B", (4, 2))]:
            params[f"layers.0.{parent}.{leaf}.lora_{side}"] = torch.nn.Parameter(torch.randn(shape))
    state = to_peft_state_dict(params, config)
    for p in params.values():
        p.data.zero_()
    from_peft_state_dict(state, params, config)
    assert all(torch.equal(v, to_peft_state_dict(params, config)[k]) for k, v in state.items())
    source = tmp_path / "adapter"
    source.mkdir()
    save_file(state, source / "adapter_model.safetensors")
    (source / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3.8-Flash-Next", "peft_type": "LORA",
        "r": 2, "lora_alpha": 4, "lora_dropout": 0, "bias": "none",
        "target_modules": config.target_modules,
    }))
    export_runtime_lora_adapter(source_adapter_dir=source, output_dir=tmp_path / "runtime")


def test_all_default_targets_map_to_their_flash_parent():
    from training.flash.dense_lora import ALL_TARGETS, DENSE_TARGETS

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.linear_attn = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.shared_expert = torch.nn.Module()
    for parent_name, leaves in DENSE_TARGETS.items():
        for leaf in leaves:
            setattr(layer.get_submodule(parent_name), leaf, torch.nn.Linear(4, 4))
    params = dense_lora_policy(r=2).configure_layer(layer, 0)
    assert len(params) == 2 * len(ALL_TARGETS)
    for parent_name, leaves in DENSE_TARGETS.items():
        for leaf in leaves:
            assert isinstance(layer.get_submodule(f"{parent_name}.{leaf}"), LoRALinear)
            assert f"{parent_name}.{leaf}.lora_A" in params
