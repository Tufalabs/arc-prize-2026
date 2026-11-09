"""CPU-only complete checkpoint header/shape and pinned source audit."""

import argparse
import json
import struct
from pathlib import Path

import torch

from sft.distillation.common import TEACHER, dump, sha, validate_student
from training.flash.adapters import validate_transformers_contract


def audit(root, experts):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    validate_transformers_contract()
    mapping = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers, inventory = {}, {}
    for name in sorted(set(mapping.values())):
        path = root / name
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            raw = f.read(length)
        header = json.loads(raw)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        assert (
            max(v["data_offsets"][1] for v in tensors.values()) + length + 8
            == path.stat().st_size
        )
        headers.update(tensors)
        import hashlib

        inventory[name] = {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
        }
    config = Qwen4ExpTextConfig(
        **json.loads((root / "config.json").read_text())["text_config"]
    )
    assert config.num_experts == experts and config.num_experts_per_tok == 10
    config._attn_implementation, config._experts_implementation = "sdpa", "grouped_mm"
    with torch.device("meta"):
        model = m.Qwen4ExpTextModel(config)
    for name, parameter in model.named_parameters():
        key = "model.language_model." + name
        if name.endswith("ngram_embedding.weight"):
            prefix = key.removesuffix(".weight")
            parts = [headers[f"{prefix}.shard_{i}.weight"] for i in range(128)]
            assert sum(v["shape"][0] for v in parts) == parameter.shape[0]
            assert all(
                v["shape"][1] == parameter.shape[1] and v["dtype"] == "F8_E4M3"
                for v in parts
            )
            assert prefix + ".weight_scale" in mapping
        elif name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            prefix = key.rsplit(".", 1)[0]
            projects = (
                ["down_proj"]
                if name.endswith("down_proj")
                else ["gate_proj", "up_proj"]
            )
            for e in range(experts):
                for proj in projects:
                    k = f"{prefix}.{e}.{proj}"
                    shape = [2560, 640] if proj == "down_proj" else [640, 2560]
                    assert headers[k + ".weight"]["shape"] == [shape[0], shape[1] // 2]
                    assert headers[k + ".weight"]["dtype"] == "U8"
                    assert headers[k + ".weight_scale"]["shape"] == [
                        shape[0],
                        shape[1] // 16,
                    ]
                    assert k + ".weight_scale_2" in mapping
        else:
            assert headers[key]["shape"] == list(parameter.shape), key
    assert headers["lm_head.weight"]["shape"] == [config.vocab_size, config.hidden_size]
    return {
        "config_sha256": sha(root / "config.json"),
        "index_sha256": sha(root / "model.safetensors.index.json"),
        "tokenizer_sha256": sha(root / "tokenizer.json"),
        "experts": experts,
        "shards": inventory,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("preflight.json"))
    args = ap.parse_args()
    if args.teacher.resolve() != TEACHER.resolve():
        raise ValueError("Teacher must be e307")
    roles = validate_student(args.teacher, args.student)
    teacher, student = audit(args.teacher, 307), audit(args.student, roles["experts"])
    assert teacher["tokenizer_sha256"] == student["tokenizer_sha256"]
    dump(
        args.out,
        {
            "status": "cpu_headers_passed_not_gpu_qualified",
            "teacher": teacher,
            "student": student,
        },
    )
    print("Teacher/student checkpoint header and shape audits passed")


if __name__ == "__main__":
    main()
