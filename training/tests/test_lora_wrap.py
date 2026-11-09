"""Unit tests for the F3 LoRA-wrap assert (sft/lora_wrap.py).

CPU-only, no model downloads: dummy nn.Module trees with real peft
``lora.Linear`` layers grafted at realistic paths, checked against a plain
``SFTRunConfig``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch.nn as nn
from peft.tuners.lora import Linear as LoraLinear

from sft.config import SFTRunConfig
from sft.lora_wrap import assert_lora_wrap

_LM = "base_model.model.model.language_model.layers"


def _build_tree(wrapped_paths: list[str]) -> nn.Module:
    """A nested module tree with a real peft LoraLayer at each dotted path."""
    root = nn.Module()
    for path in wrapped_paths:
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            child = getattr(node, part, None)
            if child is None:
                child = nn.Module()
                setattr(node, part, child)
            node = child
        setattr(
            node, parts[-1], LoraLinear(nn.Linear(4, 4), "default", r=2, lora_alpha=4)
        )
    return root


def _cfg(tmp_path: Path, **overrides) -> SFTRunConfig:
    defaults = dict(
        output_dir=str(tmp_path / "out"), lora_target_modules=["q_proj", "gate_proj"]
    )
    defaults.update(overrides)
    return SFTRunConfig(**defaults)


def test_valid_wrap_passes_and_writes_census(tmp_path: Path) -> None:
    model = _build_tree(
        [
            f"{_LM}.0.self_attn.q_proj",
            f"{_LM}.3.self_attn.q_proj",
            f"{_LM}.0.mlp.gate_proj",
            f"{_LM}.1.mlp.gate_proj",
            f"{_LM}.2.mlp.gate_proj",
        ]
    )
    cfg = _cfg(tmp_path)

    counts = assert_lora_wrap(model, cfg)

    assert counts == {"q_proj": 2, "gate_proj": 3}
    census = json.loads(
        (Path(cfg.output_dir) / "lora_wrap_census.json").read_text(encoding="utf-8")
    )
    assert census == {"counts": {"gate_proj": 3, "q_proj": 2}, "total": 5}


def test_visual_path_raises_naming_offender(tmp_path: Path) -> None:
    model = _build_tree(
        [
            f"{_LM}.0.self_attn.q_proj",
            f"{_LM}.0.mlp.gate_proj",
            "base_model.model.model.visual.blocks.0.attn.q_proj",
        ]
    )
    with pytest.raises(RuntimeError, match=r"visual"):
        assert_lora_wrap(model, _cfg(tmp_path))


def test_mtp_path_raises_naming_offender(tmp_path: Path) -> None:
    model = _build_tree(
        [
            f"{_LM}.0.self_attn.q_proj",
            f"{_LM}.0.mlp.gate_proj",
            "base_model.model.model.mtp.layers.0.self_attn.q_proj",
        ]
    )
    with pytest.raises(RuntimeError, match=r"mtp"):
        assert_lora_wrap(model, _cfg(tmp_path))


def test_non_language_model_path_raises(tmp_path: Path) -> None:
    model = _build_tree(
        [
            f"{_LM}.0.self_attn.q_proj",
            f"{_LM}.0.mlp.gate_proj",
            "base_model.model.model.layers.0.self_attn.q_proj",  # text-class path, not language_model
        ]
    )
    with pytest.raises(RuntimeError, match=r"language_model"):
        assert_lora_wrap(model, _cfg(tmp_path))


def test_suffix_mismatch_raises_missing_and_unexpected(tmp_path: Path) -> None:
    model = _build_tree(
        [
            f"{_LM}.0.self_attn.q_proj",
            f"{_LM}.0.mlp.up_proj",  # not in target_modules
        ]
    )
    with pytest.raises(
        RuntimeError, match=r"missing=\['gate_proj'\].*unexpected=\['up_proj'\]"
    ):
        assert_lora_wrap(model, _cfg(tmp_path))


def test_zero_wrapped_layers_raises(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))  # no LoraLayer anywhere
    with pytest.raises(RuntimeError, match=r"no LoRA-wrapped modules"):
        assert_lora_wrap(model, _cfg(tmp_path))


def test_disabled_lora_skips_entirely(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 4))  # would fail every check if inspected
    cfg = _cfg(tmp_path, use_lora=False)

    assert assert_lora_wrap(model, cfg) is None
    assert not (Path(cfg.output_dir) / "lora_wrap_census.json").exists()
