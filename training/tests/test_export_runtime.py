"""Unit tests for the SFT runtime-adapter export (F1/F2).

CPU-only, tiny random tensors, no model downloads. Fake adapters are built in
``tmp_path`` with a realistic PEFT ``adapter_config.json`` + safetensors payload.
The multimodal SFT namespace (``base_model.model.model.language_model.``) is the
passthrough case; the text-class namespace regression (the RL contract) lives in
``tests/test_runtime_lora.py`` and must stay green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sft.export_runtime import export_run_checkpoints
from training.runtime_lora import export_runtime_lora_adapter

_LANG = "base_model.model.model.language_model."
_ALL_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
_SUB = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}


def _base_config(**overrides) -> dict:
    config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.18.1",
        "r": 16,
        "target_modules": list(_ALL_TARGETS),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    config.update(overrides)
    return config


def _write_adapter(
    adapter_dir: Path, *, tensors: dict, config_overrides: dict | None = None
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(_base_config(**(config_overrides or {})), indent=2), encoding="utf-8"
    )
    save_file(tensors, adapter_dir / "adapter_model.safetensors")


def _passthrough_tensors(*, r: int = 16, zero_b: bool = False, layers=(0, 1)) -> dict:
    """A small multimodal-namespace payload across a couple of language-model layers."""
    tensors: dict[str, torch.Tensor] = {}
    seed = 0
    for layer in layers:
        for proj in _ALL_TARGETS:
            base = f"{_LANG}layers.{layer}.{_SUB[proj]}.{proj}"
            seed += 1
            tensors[f"{base}.lora_A.weight"] = torch.randn(r, 6, generator=None) + seed
            b = torch.zeros(5, r) if zero_b else (torch.randn(5, r) + seed)
            tensors[f"{base}.lora_B.weight"] = b
    return tensors


# --- passthrough (SFT multimodal namespace) ---------------------------------


def test_passthrough_copies_sft_namespace_keys_and_values(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    runtime = tmp_path / "checkpoint-1-runtime"
    source_tensors = _passthrough_tensors()
    _write_adapter(source, tensors=source_tensors)

    export = export_runtime_lora_adapter(
        source_adapter_dir=source,
        output_dir=runtime,
        compute_delta_stats=True,
        compute_adapter_sha256=True,
    )

    assert export.export_mode == "qwen3_multimodal_passthrough"
    runtime_tensors = load_file(runtime / "adapter_model.safetensors")
    assert runtime_tensors.keys() == source_tensors.keys()
    for key, tensor in source_tensors.items():
        assert torch.equal(runtime_tensors[key], tensor)

    summary = json.loads(
        (runtime / "runtime_adapter_summary.json").read_text(encoding="utf-8")
    )
    assert summary["export_mode"] == "qwen3_multimodal_passthrough"
    assert summary["base_model_name"] == "Qwen/Qwen3.5-9B"
    assert summary["peft_version"] == "0.18.1"
    assert summary["scaling_mode"] == "standard"
    assert summary["adapter_sha256"] == export.adapter_sha256
    assert len(summary["adapter_sha256"]) == 64
    assert set(summary["delta_stats"]) == set(_ALL_TARGETS)


def test_passthrough_rejects_visual_namespace(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    tensors = {
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight": torch.randn(
            4, 6
        ),
        "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight": torch.randn(
            5, 4
        ),
    }
    _write_adapter(source, tensors=tensors)
    with pytest.raises(ValueError, match=r"visual"):
        export_runtime_lora_adapter(
            source_adapter_dir=source, output_dir=tmp_path / "rt"
        )


def test_passthrough_rejects_mtp_namespace(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    tensors = {
        "base_model.model.model.mtp.layers.0.q_proj.lora_A.weight": torch.randn(4, 6),
        "base_model.model.model.mtp.layers.0.q_proj.lora_B.weight": torch.randn(5, 4),
    }
    _write_adapter(source, tensors=tensors)
    with pytest.raises(ValueError, match=r"mtp"):
        export_runtime_lora_adapter(
            source_adapter_dir=source, output_dir=tmp_path / "rt"
        )


def test_mixed_layout_still_rejects_vision_tower_keys(tmp_path: Path) -> None:
    # Mixed passthrough/text layouts are normalized per key (linear_attn
    # coverage), but non-language keys like visual.* stay rejected by name.
    source = tmp_path / "checkpoint-1"
    tensors = _passthrough_tensors(layers=(0,))
    tensors["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"] = (
        torch.randn(16, 6)
    )
    tensors["base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight"] = (
        torch.randn(5, 16)
    )
    _write_adapter(source, tensors=tensors)
    with pytest.raises(ValueError, match=r"visual"):
        export_runtime_lora_adapter(
            source_adapter_dir=source, output_dir=tmp_path / "rt"
        )


# --- F2 vLLM load-invariants (validate_load_invariants=True) -----------------


def test_invariant_bias_must_be_none(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(
        source, tensors=_passthrough_tensors(), config_overrides={"bias": "lora_only"}
    )
    with pytest.raises(ValueError, match=r"bias='none'"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            validate_load_invariants=True,
        )


def test_invariant_modules_to_save_must_be_empty(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(
        source,
        tensors=_passthrough_tensors(),
        config_overrides={"modules_to_save": ["embed_tokens"]},
    )
    with pytest.raises(ValueError, match=r"modules_to_save"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            validate_load_invariants=True,
        )


def test_invariant_dora_rejected(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(
        source, tensors=_passthrough_tensors(), config_overrides={"use_dora": True}
    )
    with pytest.raises(ValueError, match=r"DoRA"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            validate_load_invariants=True,
        )


def test_invariant_target_modules_must_be_subset(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(
        source,
        tensors=_passthrough_tensors(),
        config_overrides={"target_modules": ["q_proj", "mlp_experts"]},
    )
    with pytest.raises(ValueError, match=r"language-projection set"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            validate_load_invariants=True,
        )


def test_invariant_rank_off_ladder_names_allowed_values(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(source, tensors=_passthrough_tensors(), config_overrides={"r": 513})
    with pytest.raises(ValueError, match=r"supported vLLM max-lora-rank"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            validate_load_invariants=True,
        )


def test_rslora_records_scaling_mode(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(
        source, tensors=_passthrough_tensors(), config_overrides={"use_rslora": True}
    )
    export = export_runtime_lora_adapter(
        source_adapter_dir=source,
        output_dir=tmp_path / "rt",
        validate_load_invariants=True,
    )
    assert export.scaling_mode == "rslora"
    summary = json.loads(
        (tmp_path / "rt" / "runtime_adapter_summary.json").read_text(encoding="utf-8")
    )
    assert summary["scaling_mode"] == "rslora"


# --- delta stats -------------------------------------------------------------


def test_delta_stats_match_known_tensors(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    r, alpha = 2, 4.0
    a = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])  # [r=2, in=3]
    b = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 0.0], [1.0, 1.0]])  # [out=4, r=2]
    tensors = {
        f"{_LANG}layers.0.self_attn.q_proj.lora_A.weight": a,
        f"{_LANG}layers.0.self_attn.q_proj.lora_B.weight": b,
    }
    _write_adapter(
        source,
        tensors=tensors,
        config_overrides={"r": r, "lora_alpha": alpha, "target_modules": ["q_proj"]},
    )

    export = export_runtime_lora_adapter(
        source_adapter_dir=source, output_dir=tmp_path / "rt", compute_delta_stats=True
    )

    delta = (b @ a) * (alpha / r)
    stats = export.delta_stats["q_proj"]
    assert stats["count"] == 1
    assert stats["max_abs_delta"] == pytest.approx(float(delta.abs().max()))
    assert stats["mean_abs_delta"] == pytest.approx(float(delta.abs().mean()))
    assert stats["frac_B_nonzero"] == pytest.approx(float(b.ne(0).float().mean()))


# --- zero-delta gate ---------------------------------------------------------


def test_zero_b_rejected_without_flag(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(source, tensors=_passthrough_tensors(zero_b=True))
    with pytest.raises(ValueError, match=r"all-zero lora_B"):
        export_runtime_lora_adapter(
            source_adapter_dir=source,
            output_dir=tmp_path / "rt",
            allow_zero_delta=False,
        )


def test_zero_b_allowed_with_flag(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-1"
    _write_adapter(source, tensors=_passthrough_tensors(zero_b=True))
    export = export_runtime_lora_adapter(
        source_adapter_dir=source,
        output_dir=tmp_path / "rt",
        allow_zero_delta=True,
        compute_delta_stats=True,
    )
    assert export.delta_stats
    for stats in export.delta_stats.values():
        assert stats["frac_B_nonzero"] == 0.0


# --- CLI discovery + idempotency ---------------------------------------------


def test_cli_exports_then_skips_when_current(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "outputs" / "checkpoint-5"
    _write_adapter(checkpoint, tensors=_passthrough_tensors())

    first = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    assert [r.status for r in first] == ["exported"]
    runtime_dir = first[0].runtime_dir
    assert runtime_dir == run_dir / "outputs" / "checkpoint-5-runtime"
    sha_after_first = json.loads(
        (runtime_dir / "runtime_adapter_summary.json").read_text()
    )["adapter_sha256"]

    second = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    assert [r.status for r in second] == ["skipped"]
    sha_after_second = json.loads(
        (runtime_dir / "runtime_adapter_summary.json").read_text()
    )["adapter_sha256"]
    assert sha_after_first == sha_after_second


def test_rerun_reexports_when_runtime_artifact_missing(tmp_path: Path) -> None:
    # An incomplete runtime dir (e.g. adapter_config.json lost) must not be
    # reported "skipped" -- vLLM cannot load it.
    run_dir = tmp_path / "run"
    _write_adapter(run_dir / "outputs" / "checkpoint-5", tensors=_passthrough_tensors())
    [first] = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    (first.runtime_dir / "adapter_config.json").unlink()

    [second] = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    assert second.status == "exported"
    assert (second.runtime_dir / "adapter_config.json").is_file()


def test_rerun_reexports_when_runtime_config_was_edited(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_adapter(run_dir / "outputs" / "checkpoint-5", tensors=_passthrough_tensors())
    [first] = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    runtime_config_path = first.runtime_dir / "adapter_config.json"
    runtime_config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    runtime_config["lora_alpha"] = 999
    runtime_config_path.write_text(json.dumps(runtime_config), encoding="utf-8")

    [second] = export_run_checkpoints(run_dir=run_dir, checkpoint=5)

    assert second.status == "exported"
    assert (
        json.loads(runtime_config_path.read_text(encoding="utf-8"))["lora_alpha"] == 32
    )


def test_rerun_reexports_when_source_changes(tmp_path: Path) -> None:
    # Editing the source checkpoint (config or weights) invalidates the
    # runtime export; a rerun must re-export, not retain the stale artifacts.
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "outputs" / "checkpoint-5"
    _write_adapter(checkpoint, tensors=_passthrough_tensors())
    export_run_checkpoints(run_dir=run_dir, checkpoint=5)

    _write_adapter(
        checkpoint, tensors=_passthrough_tensors(), config_overrides={"lora_alpha": 16}
    )
    [rerun] = export_run_checkpoints(run_dir=run_dir, checkpoint=5)
    assert rerun.status == "exported"
    runtime_config = json.loads((rerun.runtime_dir / "adapter_config.json").read_text())
    assert runtime_config["lora_alpha"] == 16


def test_cli_requires_checkpoint_or_all(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_adapter(run_dir / "outputs" / "checkpoint-5", tensors=_passthrough_tensors())
    with pytest.raises(SystemExit, match=r"--checkpoint N or --all"):
        export_run_checkpoints(run_dir=run_dir)


def test_api_rejects_checkpoint_and_all_together(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_adapter(run_dir / "outputs" / "checkpoint-5", tensors=_passthrough_tensors())
    with pytest.raises(ValueError, match="mutually exclusive"):
        export_run_checkpoints(run_dir=run_dir, checkpoint=5, export_all=True)
