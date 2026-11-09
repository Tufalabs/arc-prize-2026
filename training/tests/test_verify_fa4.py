import json
import sys
import types
from importlib import metadata
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
import torch

from training.verify_fa4 import (
    EXPECTED_FA4_VERSION,
    _fa4_compatible_input,
    verify_fa4,
)


def test_all_ppo_configs_select_fa4_explicitly() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    for config_path in config_dir.glob("ppo_*.json"):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert payload["training"]["attn_implementation"] == "flash_attention_4"


def test_verify_fa4_rejects_wrong_version(monkeypatch) -> None:
    metadata_version = metadata.version
    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: "4.0.0b18" if name == "flash-attn-4" else metadata_version(name),
    )

    with pytest.raises(RuntimeError, match="unexpected flash-attn-4 version"):
        verify_fa4()


def test_verify_fa4_requires_cuda_after_import_checks(monkeypatch) -> None:
    cute_module = types.ModuleType("flash_attn.cute")
    cute_module.flash_attn_func = object()
    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []
    flash_attn_module.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn_module)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute_module)
    monkeypatch.setattr(
        "training.verify_fa4.require_fa4_distribution",
        lambda: EXPECTED_FA4_VERSION,
    )
    monkeypatch.setattr("training.verify_fa4.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        verify_fa4()


def test_fa4_compatible_input_materializes_qwen_gated_query_stride() -> None:
    query_and_gate = torch.randn(1, 16, 24, 512)
    query = query_and_gate.chunk(2, dim=-1)[0].transpose(1, 2)

    result = _fa4_compatible_input(query)

    assert result.transpose(1, 2).is_contiguous()
    torch.testing.assert_close(result, query)
