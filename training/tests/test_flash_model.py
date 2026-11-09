"""CPU checks for the Flash-Next training pieces that need no checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from training import flash_model
from training.frozen_nvfp4 import (
    FrozenNVFP4Expert,
    FrozenNVFP4Experts,
    FrozenNVFP4Linear,
    dequantize_nvfp4,
)


def _flash_config(path: Path, *, experts: int = 512) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "quantization_config": {"quant_algo": "NVFP4"},
                "text_config": {
                    "num_hidden_layers": 48,
                    "num_experts_per_tok": 10,
                    "hidden_size": 2560,
                    "num_experts": experts,
                },
            }
        )
    )


def test_checkpoint_config_accepts_the_release_and_its_prunes(tmp_path: Path) -> None:
    for experts in (256, 307, 512):
        directory = tmp_path / f"e{experts}"
        directory.mkdir()
        _flash_config(directory, experts=experts)
        assert flash_model.flash_checkpoint_config(directory)["text_config"]["num_experts"] == experts


@pytest.mark.parametrize("experts", [10, 513])
def test_checkpoint_config_rejects_other_expert_counts(tmp_path: Path, experts: int) -> None:
    _flash_config(tmp_path, experts=experts)
    with pytest.raises(ValueError, match="unsupported Flash shape"):
        flash_model.flash_checkpoint_config(tmp_path)


def test_checkpoint_names_map_to_the_text_model() -> None:
    assert (
        flash_model.checkpoint_to_text_name("model.language_model.layers.3.self_attn.q_proj.weight")
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert flash_model.checkpoint_to_text_name("lm_head.weight") == "lm_head.weight"
    assert flash_model.checkpoint_to_text_name("model.visual.blocks.0.attn.qkv.weight") is None
    assert flash_model.checkpoint_to_text_name("mtp.layers.0.mlp.gate.weight") is None
    with pytest.raises(ValueError):
        flash_model.checkpoint_to_text_name("unexpected.weight")


def test_lora_targets_are_the_served_projections_only() -> None:
    with pytest.raises(ValueError, match="nonempty subset"):
        flash_model.attach_flash_lora(nn.Linear(2, 2), r=2, alpha=4, dropout=0.0, target_modules=["gate_proj"])
    with pytest.raises(ValueError, match="nonempty subset"):
        flash_model.attach_flash_lora(nn.Linear(2, 2), r=2, alpha=4, dropout=0.0, target_modules=[])


class _Decoder(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        assert attention_mask.shape == input_ids.shape and use_cache is False and return_dict is True

        class _Output:
            last_hidden_state = self.embed(input_ids)

        return _Output()


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 11, hidden: int = 6) -> None:
        super().__init__()
        self.model = _Decoder(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


def test_selected_token_logprobs_matches_full_log_softmax() -> None:
    torch.manual_seed(0)
    model = _TinyCausalLM()
    ids = torch.randint(0, 11, (1, 9))
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 4:] = True
    scored = flash_model.selected_token_logprobs(model, ids, mask, head_chunk=2)
    hidden = model.model.embed(ids)[:, :-1]
    reference = torch.log_softmax(model.lm_head(hidden).float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    assert scored.shape == (1, 8)
    assert torch.allclose(scored[mask[:, 1:]], reference[mask[:, 1:]], atol=1e-6)
    assert torch.all(scored[~mask[:, 1:]] == 0)


def test_frozen_nvfp4_linear_dequantises_and_backpropagates_to_inputs() -> None:
    torch.manual_seed(0)
    out_features, in_features = 4, 32
    weight = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scale = torch.rand(out_features, in_features // 16).to(torch.float8_e4m3fn)
    scale2 = torch.tensor(0.5, dtype=torch.float32)
    layer = FrozenNVFP4Linear(weight, scale, scale2, compute_dtype=torch.float32)
    assert (layer.in_features, layer.out_features) == (in_features, out_features)
    dense = dequantize_nvfp4(weight, scale, scale2, dtype=torch.float32)
    x = torch.randn(3, in_features, requires_grad=True)
    y = layer(x)
    assert torch.allclose(y, x @ dense.T, atol=1e-5)
    y.sum().backward()
    assert x.grad is not None and torch.allclose(x.grad, dense.sum(0).expand_as(x), atol=1e-5)
    assert not any(p.requires_grad for p in layer.parameters())


def _frozen_linear(out_features: int, in_features: int, seed: int) -> FrozenNVFP4Linear:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8, generator=generator)
    scale = torch.rand(out_features, in_features // 16, generator=generator).to(torch.float8_e4m3fn)
    return FrozenNVFP4Linear(weight, scale, torch.tensor(0.25), compute_dtype=torch.float32)


def test_frozen_experts_dispatch_matches_a_dense_reference_and_saves_no_expert_copies() -> None:
    torch.manual_seed(1)
    hidden, inter, experts, tokens, top_k = 32, 16, 5, 7, 2
    bank = FrozenNVFP4Experts(
        [
            FrozenNVFP4Expert(
                _frozen_linear(inter, hidden, 3 * e),
                _frozen_linear(inter, hidden, 3 * e + 1),
                _frozen_linear(hidden, inter, 3 * e + 2),
            )
            for e in range(experts)
        ]
    )
    x = torch.randn(tokens, hidden, requires_grad=True)
    index = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k, requires_grad=True)
    out = bank(x, index, weights)
    reference = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(top_k):
            reference[t] += weights[t, k] * bank[int(index[t, k])](x[t : t + 1])[0]
    assert torch.allclose(out, reference, atol=1e-4)
    out.sum().backward()
    assert x.grad is not None and weights.grad is not None
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()
    assert not any(p.requires_grad for p in bank.parameters())


class _FakeReader:
    """Checkpoint keys for one layer of two tiny experts, generated on demand."""

    def __init__(self, hidden: int, inter: int, experts: int, layer: int) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        generator = torch.Generator().manual_seed(7)
        for expert in range(experts):
            for projection, (out_features, in_features) in (
                ("gate_proj", (inter, hidden)),
                ("up_proj", (inter, hidden)),
                ("down_proj", (hidden, inter)),
            ):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
                self.tensors[prefix + ".weight"] = torch.randint(
                    0, 256, (out_features, in_features // 2), dtype=torch.uint8, generator=generator
                )
                self.tensors[prefix + ".weight_scale"] = torch.rand(
                    out_features, in_features // 16, generator=generator
                ).to(torch.float8_e4m3fn)
                self.tensors[prefix + ".weight_scale_2"] = torch.rand(1, generator=generator)
                self.tensors[prefix + ".input_scale"] = torch.ones(1)
        self.weight_map = {key: "shard" for key in self.tensors}

    def tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]


def test_expert_bank_loads_every_expert_from_stacked_copies() -> None:
    reader = _FakeReader(hidden=32, inter=16, experts=3, layer=5)
    bank, consumed = flash_model.load_expert_bank(reader, 5, 3, torch.device("cpu"))
    assert consumed == set(reader.tensors)
    assert bank.num_experts == 3 and (bank.hidden_dim, bank.intermediate_dim) == (32, 16)
    for expert in range(3):
        prefix = f"model.language_model.layers.5.mlp.experts.{expert}"
        assert torch.equal(bank[expert].up_proj.weight, reader.tensors[f"{prefix}.up_proj.weight"])
        assert torch.equal(
            bank[expert].down_proj.weight_scale.view(torch.uint8),
            reader.tensors[f"{prefix}.down_proj.weight_scale"].view(torch.uint8),
        )
        assert float(bank[expert].gate_proj.weight_scale_2) == float(reader.tensors[f"{prefix}.gate_proj.weight_scale_2"])
    x = torch.randn(4, 32)
    out = bank(x, torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]]), torch.full((4, 2), 0.5))
    assert out.shape == (4, 32) and torch.isfinite(out).all()


def test_host_ple_table_stays_on_the_host_and_accepts_any_index_device() -> None:
    shards = [torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.float8_e4m3fn), torch.ones(2, 4).to(torch.float8_e4m3fn)]
    table = flash_model.FrozenHostShardedEmbedding(shards, torch.tensor(2.0))
    assert "weight" not in dict(table.named_buffers()) and table.weight.device.type == "cpu"
    table.to("cpu")  # a whole-model .to() must not move the placeholder
    rows = table(torch.tensor([[0, 4], [2, 3]]))
    assert rows.shape == (2, 2, 4) and rows.dtype == torch.bfloat16
    assert torch.allclose(rows[1, 1].float(), torch.full((4,), 2.0))
    assert torch.allclose(rows[0, 1].float(), torch.full((4,), 2.0))
    with pytest.raises(IndexError):
        table(torch.tensor([5]))


def _tiny_flash_text_config() -> dict:
    return {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "head_dim": 4,
        "num_key_value_heads": 1,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 4,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 4,
        "layer_types": ["linear_attention", "full_attention", "linear_attention"],
    }


def test_synthetic_flash_adapter_matches_the_trainer_layout_and_exports(tmp_path):
    from safetensors.torch import load_file

    from training.flash_model import (
        FLASH_LORA_TARGET_MODULES,
        flash_lora_shapes,
        write_flash_peft_adapter,
    )
    from training.runtime_lora import export_runtime_lora_adapter

    text_config = _tiny_flash_text_config()
    attention, gdn = flash_lora_shapes(text_config)
    assert attention["q_proj"] == (16, 2 * 4 * 2)  # query + sigmoid gate
    assert attention["k_proj"] == (16, 4)
    assert gdn["in_proj_qkv"] == (16, 8 * 2 + 8)
    assert gdn["out_proj"] == (8, 16)

    source = tmp_path / "peft"
    count = write_flash_peft_adapter(
        source,
        checkpoint="/ckpt/Qwen3.8-Flash-Next-NVFP4-test",
        text_config=text_config,
        rank=2,
        alpha=4.0,
    )
    # Two GDN layers x 3 targets + one attention layer x 4 targets, A and B each.
    assert count == (2 * 3 + 1 * 4) * 2
    tensors = load_file(str(source / "adapter_model.safetensors"))
    assert "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight" in tensors
    assert "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight" in tensors
    assert tensors["base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"].shape == (2, 16)
    assert tensors["base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"].shape == (24, 2)
    # Identity: every lora_B is zero, every lora_A is a real (nonzero) init.
    for name, tensor in tensors.items():
        if name.endswith("lora_B.weight"):
            assert not tensor.any()
        else:
            assert tensor.any()
    config = json.loads((source / "adapter_config.json").read_text())
    assert config["r"] == 2 and config["lora_alpha"] == 4.0
    assert config["target_modules"] == list(FLASH_LORA_TARGET_MODULES)

    export = export_runtime_lora_adapter(source_adapter_dir=source, output_dir=tmp_path / "runtime")
    assert export.converted_tensor_count == count
    runtime = load_file(str(tmp_path / "runtime" / "adapter_model.safetensors"))
    assert "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight" in runtime


def test_synthetic_flash_adapter_honours_a_target_subset(tmp_path):
    from training.flash_model import write_flash_peft_adapter

    count = write_flash_peft_adapter(
        tmp_path / "peft",
        checkpoint="/ckpt/Qwen3.8-Flash-Next-NVFP4-test",
        text_config=_tiny_flash_text_config(),
        rank=2,
        alpha=4.0,
        target_modules=["q_proj"],
    )
    assert count == 2  # one attention layer, A and B
    with pytest.raises(ValueError, match="target_modules"):
        write_flash_peft_adapter(
            tmp_path / "bad",
            checkpoint="/ckpt/Qwen3.8-Flash-Next-NVFP4-test",
            text_config=_tiny_flash_text_config(),
            rank=2,
            alpha=4.0,
            target_modules=["router"],
        )


@pytest.mark.parametrize("targets,expected", [
    (("gate_proj", "up_proj", "down_proj"), 18),
    (("in_proj_b", "in_proj_a"), 8),
])
def test_optional_flash_targets_have_dense_shapes_and_export(tmp_path, targets, expected):
    from safetensors.torch import load_file
    from training.runtime_lora import export_runtime_lora_adapter

    config = _tiny_flash_text_config() | {"shared_expert_intermediate_size": 24}
    source = tmp_path / "peft"
    count = flash_model.write_flash_peft_adapter(
        source, checkpoint="/ckpt/Qwen3.8-Flash-Next", text_config=config,
        rank=2, alpha=4, b_scale=0.02, target_modules=targets,
    )
    assert count == expected
    exported = export_runtime_lora_adapter(
        source_adapter_dir=source, output_dir=tmp_path / "runtime", validate_load_invariants=True,
    )
    tensors = load_file(str(exported.adapter_dir / "adapter_model.safetensors"))
    for name, tensor in tensors.items():
        assert "model.language_model.layers." in name and ".experts." not in name
        if "shared_expert" in name:
            fan_in, fan_out = (24, 16) if "down_proj" in name else (16, 24)
        else:
            fan_in, fan_out = 16, 2
        assert tensor.shape == ((2, fan_in) if "lora_A" in name else (fan_out, 2))
        assert tensor.any()


def test_flash_probe_scale_changes_only_b(tmp_path):
    from safetensors.torch import load_file

    targets = flash_model.FLASH_LORA_TARGET_MODULES + flash_model.FLASH_SHARED_EXPERT_LORA_TARGETS
    config = _tiny_flash_text_config() | {"shared_expert_intermediate_size": 24}
    for name, scale in (("large", 0.02), ("small", 0.005)):
        flash_model.write_flash_peft_adapter(
            tmp_path / name, checkpoint="synthetic", text_config=config,
            rank=16, alpha=32, b_scale=scale, seed=17, target_modules=targets,
        )
    large = load_file(tmp_path / "large/adapter_model.safetensors")
    small = load_file(tmp_path / "small/adapter_model.safetensors")
    for name, value in large.items():
        expected = value * 0.25 if name.endswith("lora_B.weight") else value
        assert torch.equal(small[name], expected)
