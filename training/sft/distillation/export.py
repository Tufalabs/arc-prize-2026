"""Export a distillation checkpoint with expert LoRA merged into BF16 weights.

Usage: python -m sft.distillation.export --student BASE --checkpoint STATE --out NEW
The destination is published only after every trainable tensor has been consumed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from training.flash.expert_lora import merge_projection, normalize_lora
from training.flash.reader import Reader

from .common import sha

PREFIX = "model.language_model."
EXPERT = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.weight$"
)
EXPERT_AUX = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate|up|down)_proj\.(?:input_scale|weight_scale|weight_scale_2)$"
)


def export_checkpoint(student, checkpoint, out, *, device="cpu"):
    student, checkpoint, out = map(Path, (student, checkpoint, out))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    manifest = state["manifest"]
    lora = normalize_lora(manifest.get("expert_lora"))
    if lora is None:
        raise ValueError("this exporter requires an expert-LoRA checkpoint")
    for field, filename in (
        ("student_config_sha256", "config.json"),
        ("student_index_sha256", "model.safetensors.index.json"),
        ("pruning_sha256", "distillation-parent.json"),
    ):
        if manifest.get(field) != sha(student / filename):
            raise ValueError(f"checkpoint source identity mismatch: {filename}")
    trainables = state["routers"]  # legacy checkpoint field includes adapters
    if state["router_names"] != list(trainables):
        raise ValueError("checkpoint trainable census mismatch")
    if not all(
        v.dtype == torch.float32 and torch.isfinite(v).all()
        for v in trainables.values()
    ):
        raise ValueError("trainables must be finite FP32 tensors")
    config = json.loads((student / "config.json").read_text())
    text_config = config["text_config"]
    layers, experts = text_config["num_hidden_layers"], text_config["num_experts"]
    expected = {
        f"layers.{layer}.mlp.experts.lora_{projection}_{side}"
        for layer in range(layers)
        for projection in ("gate", "up", "down")
        for side in ("A", "B")
    }
    if manifest.get("train_routers", True):
        expected.update(f"layers.{layer}.mlp.gate.weight" for layer in range(layers))
    if set(trainables) != expected:
        raise ValueError(
            "checkpoint must contain exactly the configured routers and expert adapters"
        )
    for section in (config, text_config):
        quant = section.pop("quantization_config", None)
        if quant and quant.get("quant_algo") != "NVFP4":
            raise ValueError("unsupported source quantization configuration")
    reader = Reader(student)
    index = json.loads((student / "model.safetensors.index.json").read_text())
    shards = {}
    for key, shard in index["weight_map"].items():
        if Path(shard).name != shard:
            raise ValueError("source shard must be a filename")
        shards.setdefault(shard, []).append(key)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    consumed, new_map, total_size, merged = set(), {}, 0, 0
    try:
        for shard, keys in sorted(shards.items()):
            tensors = {}
            for key in keys:
                if EXPERT_AUX.match(key):
                    continue
                match = EXPERT.match(key)
                short = key.removeprefix(PREFIX)
                if match:
                    layer, expert, projection = match.groups()
                    expert = int(expert)
                    if int(layer) >= layers or expert >= experts:
                        raise ValueError("source expert index out of bounds")
                    weight = reader.dequant(key.removesuffix(".weight"), device)
                    names = [
                        f"layers.{layer}.mlp.experts.lora_{projection}_{side}"
                        for side in ("A", "B")
                    ]
                    a, b = (trainables[name] for name in names)
                    if a.shape[0] != experts or b.shape[0] != experts:
                        raise ValueError("adapter expert count mismatch")
                    value = merge_projection(
                        weight, a[expert].to(device), b[expert].to(device), lora
                    ).cpu()
                    consumed.update(names)
                    merged += 1
                elif short in trainables:
                    value = trainables[short]
                    if value.shape != reader.tensor(key).shape:
                        raise ValueError("router shape mismatch")
                    consumed.add(short)
                else:
                    value = reader.tensor(key)
                tensors[key] = value.contiguous()
                total_size += value.numel() * value.element_size()
                new_map[key] = shard
            save_file(tensors, temporary / shard)
        if consumed != expected or merged != layers * experts * 3:
            raise ValueError("incomplete adapter merge")
        excluded = set(shards) | {
            "config.json",
            "model.safetensors.index.json",
            "hf_quant_config.json",
            "quantize_config.json",
        }
        for path in student.iterdir():
            if (
                path.is_file()
                and path.name not in excluded
                and path.suffix != ".safetensors"
            ):
                shutil.copy2(path, temporary / path.name)
        config["torch_dtype"] = "bfloat16"
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        index["weight_map"] = new_map
        index.setdefault("metadata", {})["total_size"] = total_size
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        provenance = {
            "source": str(student.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha(checkpoint),
            "step": state["step"],
            "expert_lora": lora,
            "merged_expert_weights": merged,
            "format": "bf16-experts-with-merged-lora-v1",
        }
        (temporary / "expert-lora-export.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        if out.exists():
            raise FileExistsError(out)
        temporary.rename(out)
        return provenance
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(
        json.dumps(
            export_checkpoint(
                args.student, args.checkpoint, args.out, device=args.device
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
