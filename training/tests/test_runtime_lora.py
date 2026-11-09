import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import training.runtime_lora as runtime_lora_module
from training.runtime_lora import (
    export_runtime_lora_adapter,
    runtime_lora_max_rank,
)


def _write_source_adapter(
    adapter_dir: Path, *, base_model_name: str = "Qwen/Qwen3.5-9B"
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model_name,
                "bias": "none",
                "inference_mode": True,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": 16,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "task_type": "CAUSAL_LM",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tensors = {
        "base_model.model.model.layers.11.self_attn.q_proj.lora_A.weight": torch.arange(
            16 * 5, dtype=torch.float32
        ).reshape(16, 5),
        "base_model.model.model.layers.11.self_attn.q_proj.lora_B.weight": torch.arange(
            7 * 16, dtype=torch.float32
        ).reshape(7, 16),
        "base_model.model.model.layers.11.self_attn.k_proj.lora_A.weight": torch.arange(
            16 * 5, dtype=torch.float32
        ).reshape(16, 5)
        + 1000,
        "base_model.model.model.layers.11.self_attn.k_proj.lora_B.weight": torch.arange(
            3 * 16, dtype=torch.float32
        ).reshape(3, 16)
        + 2000,
        "base_model.model.model.layers.11.self_attn.v_proj.lora_A.weight": torch.arange(
            16 * 5, dtype=torch.float32
        ).reshape(16, 5)
        + 3000,
        "base_model.model.model.layers.11.self_attn.v_proj.lora_B.weight": torch.arange(
            4 * 16, dtype=torch.float32
        ).reshape(4, 16)
        + 4000,
        "base_model.model.model.layers.11.self_attn.o_proj.lora_A.weight": torch.arange(
            16 * 6, dtype=torch.float32
        ).reshape(16, 6)
        + 5000,
        "base_model.model.model.layers.11.self_attn.o_proj.lora_B.weight": torch.arange(
            8 * 16, dtype=torch.float32
        ).reshape(8, 16)
        + 6000,
        "base_model.model.model.layers.11.mlp.gate_proj.lora_A.weight": torch.arange(
            16 * 6, dtype=torch.float32
        ).reshape(16, 6)
        + 7000,
        "base_model.model.model.layers.11.mlp.gate_proj.lora_B.weight": torch.arange(
            9 * 16, dtype=torch.float32
        ).reshape(9, 16)
        + 8000,
        "base_model.model.model.layers.11.mlp.up_proj.lora_A.weight": torch.arange(
            16 * 6, dtype=torch.float32
        ).reshape(16, 6)
        + 9000,
        "base_model.model.model.layers.11.mlp.up_proj.lora_B.weight": torch.arange(
            10 * 16, dtype=torch.float32
        ).reshape(10, 16)
        + 10000,
        "base_model.model.model.layers.11.mlp.down_proj.lora_A.weight": torch.arange(
            16 * 7, dtype=torch.float32
        ).reshape(16, 7)
        + 11000,
        "base_model.model.model.layers.11.mlp.down_proj.lora_B.weight": torch.arange(
            11 * 16, dtype=torch.float32
        ).reshape(11, 16)
        + 12000,
    }
    save_file(tensors, adapter_dir / "adapter_model.safetensors")


@pytest.mark.parametrize(
    "base_model_name",
    ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.6-27B"],
)
def test_export_runtime_lora_adapter_rewrites_qwen3_tensor_names_for_vllm(
    tmp_path: Path, base_model_name: str
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir, base_model_name=base_model_name)

    export = export_runtime_lora_adapter(
        source_adapter_dir=source_dir,
        output_dir=runtime_dir,
    )

    runtime_config = json.loads(
        (runtime_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    runtime_summary = json.loads(
        (runtime_dir / "runtime_adapter_summary.json").read_text(encoding="utf-8")
    )
    source_tensors = load_file(source_dir / "adapter_model.safetensors")
    runtime_tensors = load_file(runtime_dir / "adapter_model.safetensors")

    assert export.runtime_rank == 16
    assert export.runtime_target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert runtime_config["r"] == 16
    assert runtime_config["lora_alpha"] == 32
    assert runtime_config["target_modules"] == export.runtime_target_modules
    assert runtime_summary["export_mode"] == "qwen3_vllm_namespace_rewrite"
    assert runtime_summary["runtime_target_modules"] == export.runtime_target_modules
    assert runtime_summary["runtime_rank"] == 16
    assert runtime_summary["runtime_lora_alpha"] == 32.0
    assert runtime_summary["converted_tensor_count"] == len(source_tensors)
    assert runtime_tensors.keys() != source_tensors.keys()
    for key, tensor in source_tensors.items():
        runtime_key = key.replace(
            "base_model.model.model.",
            "base_model.model.model.language_model.",
            1,
        )
        assert runtime_key in runtime_tensors
        assert torch.equal(runtime_tensors[runtime_key], tensor)


def test_export_runtime_lora_adapter_handles_linear_attn_modules(
    tmp_path: Path,
) -> None:
    # Gated-DeltaNet coverage (48 of 64 Qwen3.6-27B token mixers): the
    # export must carry linear_attn.in_proj_qkv / in_proj_z / out_proj both
    # in the text-only trainer layout (model.layers.*, rewritten to the vLLM
    # language_model namespace) and in the multimodal trainer layout that is
    # ALREADY under model.language_model.* (passed through, ff22b97).
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    source_dir.mkdir(parents=True)
    (source_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.6-27B",
                "lora_alpha": 32,
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["in_proj_qkv", "in_proj_z", "out_proj"],
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "base_model.model.model.layers.2.linear_attn.in_proj_qkv.lora_A.weight": torch.ones(
            16, 5
        ),
        "base_model.model.model.layers.2.linear_attn.in_proj_qkv.lora_B.weight": torch.ones(
            10, 16
        ),
        "base_model.model.model.layers.2.linear_attn.in_proj_z.lora_A.weight": torch.ones(
            16, 5
        ),
        "base_model.model.model.layers.2.linear_attn.in_proj_z.lora_B.weight": torch.ones(
            6, 16
        ),
        "base_model.model.model.language_model.layers.3.linear_attn.out_proj.lora_A.weight": torch.ones(
            16, 6
        ),
        "base_model.model.model.language_model.layers.3.linear_attn.out_proj.lora_B.weight": torch.ones(
            5, 16
        ),
    }
    save_file(tensors, source_dir / "adapter_model.safetensors")

    export = export_runtime_lora_adapter(
        source_adapter_dir=source_dir,
        output_dir=runtime_dir,
    )

    runtime_tensors = load_file(runtime_dir / "adapter_model.safetensors")
    assert export.runtime_target_modules == ["in_proj_qkv", "in_proj_z", "out_proj"]
    assert set(runtime_tensors) == {
        "base_model.model.model.language_model.layers.2.linear_attn.in_proj_qkv.lora_A.weight",
        "base_model.model.model.language_model.layers.2.linear_attn.in_proj_qkv.lora_B.weight",
        "base_model.model.model.language_model.layers.2.linear_attn.in_proj_z.lora_A.weight",
        "base_model.model.model.language_model.layers.2.linear_attn.in_proj_z.lora_B.weight",
        "base_model.model.model.language_model.layers.3.linear_attn.out_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.3.linear_attn.out_proj.lora_B.weight",
    }


def test_export_runtime_lora_adapter_skips_checksum_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir)

    def _unexpected_checksum(_path: Path) -> str:
        pytest.fail("RL-default runtime export must not reread the adapter for SHA-256")

    monkeypatch.setattr(runtime_lora_module, "_file_sha256", _unexpected_checksum)

    export = export_runtime_lora_adapter(
        source_adapter_dir=source_dir,
        output_dir=runtime_dir,
    )
    runtime_summary = json.loads(
        (runtime_dir / "runtime_adapter_summary.json").read_text(encoding="utf-8")
    )

    assert export.adapter_sha256 == ""
    assert runtime_summary["adapter_sha256"] == ""


def test_export_runtime_lora_adapter_rejects_non_qwen3_models(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir)
    payload = json.loads(
        (source_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    payload["base_model_name_or_path"] = "meta-llama/Llama-3.1-8B"
    (source_dir / "adapter_config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="only supports Qwen 3.x adapters"):
        export_runtime_lora_adapter(
            source_adapter_dir=source_dir,
            output_dir=runtime_dir,
        )


def test_export_runtime_lora_adapter_rejects_incomplete_tensor_pair(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir)
    tensors = load_file(source_dir / "adapter_model.safetensors")
    tensors.pop("base_model.model.model.layers.11.self_attn.q_proj.lora_B.weight")
    save_file(tensors, source_dir / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="incomplete tensor pair.*missing lora_B"):
        export_runtime_lora_adapter(
            source_adapter_dir=source_dir,
            output_dir=runtime_dir,
        )

    assert not runtime_dir.exists()


def test_export_runtime_lora_adapter_rejects_tensor_rank_mismatch(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir)
    tensors = load_file(source_dir / "adapter_model.safetensors")
    tensors["base_model.model.model.layers.11.self_attn.q_proj.lora_A.weight"] = (
        torch.ones(8, 5)
    )
    tensors["base_model.model.model.layers.11.self_attn.q_proj.lora_B.weight"] = (
        torch.ones(7, 8)
    )
    save_file(tensors, source_dir / "adapter_model.safetensors")

    with pytest.raises(ValueError, match="tensor rank does not match.*configured r=16"):
        export_runtime_lora_adapter(
            source_adapter_dir=source_dir,
            output_dir=runtime_dir,
        )

    assert not runtime_dir.exists()


def test_export_runtime_lora_adapter_rejects_configured_module_without_tensors(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    runtime_dir = tmp_path / "runtime"
    _write_source_adapter(source_dir)
    config_path = source_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"].append("in_proj_z")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="target_modules have no matching tensors.*in_proj_z",
    ):
        export_runtime_lora_adapter(
            source_adapter_dir=source_dir,
            output_dir=runtime_dir,
        )

    assert not runtime_dir.exists()


def test_runtime_lora_max_rank_rounds_up_to_supported_vllm_value() -> None:
    assert (
        runtime_lora_max_rank(
            source_rank=8,
            source_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        == 8
    )
    assert (
        runtime_lora_max_rank(
            source_rank=9,
            source_target_modules=["q_proj", "k_proj", "v_proj"],
        )
        == 16
    )
