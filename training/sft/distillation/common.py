import hashlib
import json
from pathlib import Path

import torch

TEACHER = Path(
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform"
)


def teacher_identity(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())["text_config"]
    if config["num_experts"] != 307:
        raise ValueError("The teacher must be the 307-expert checkpoint")
    mapping = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    return {
        "experts": 307,
        "config_sha256": sha(root / "config.json"),
        "index_sha256": sha(root / "model.safetensors.index.json"),
        "tokenizer_sha256": sha(root / "tokenizer.json"),
        "pruning_sha256": sha(root / "pruning.json"),
        "shards": {
            n: {
                "bytes": (root / n).stat().st_size,
                "mtime_ns": (root / n).stat().st_mtime_ns,
            }
            for n in sorted(set(mapping.values()))
        },
    }


def validate_student(teacher, student):
    if not student:
        raise ValueError(
            "Student checkpoint is not configured: prune the 307-expert teacher first"
        )
    teacher, student = Path(teacher), Path(student)
    if teacher.resolve() == student.resolve():
        raise ValueError("Teacher and student must be different checkpoints")
    parent = teacher_identity(teacher)
    provenance = json.loads((student / "distillation-parent.json").read_text())
    if provenance["teacher_identity"] != parent:
        raise ValueError("Student is not derived from the pinned 307-expert teacher")
    tc = json.loads((teacher / "config.json").read_text())["text_config"]
    sc = json.loads((student / "config.json").read_text())["text_config"]
    experts = sc["num_experts"]
    if not tc["num_experts_per_tok"] <= experts < 307:
        raise ValueError("Student expert count must be below 307 and at least top-k")
    for key in set(tc) | set(sc):
        if key not in {"num_experts", "mtp_num_experts"} and tc.get(key) != sc.get(key):
            raise ValueError("Student architecture changed beyond pruning: " + key)
    if sha(student / "tokenizer.json") != parent["tokenizer_sha256"]:
        raise ValueError("Student tokenizer differs")
    kept = provenance["kept_teacher_expert_ids"]
    if set(kept) != {str(i) for i in range(tc["num_hidden_layers"])}:
        raise ValueError("Incomplete layer/expert mapping")
    for ids in kept.values():
        if (
            len(ids) != experts
            or len(set(ids)) != experts
            or any(type(i) is not int or not 0 <= i < 307 for i in ids)
        ):
            raise ValueError("Invalid ordered retained-expert mapping")
    return {
        "experts": experts,
        "teacher_identity": parent,
        "kept_teacher_expert_ids": kept,
    }


def validate_roles(config):
    if Path(config["teacher"]).resolve() != TEACHER.resolve():
        raise ValueError("Configured teacher must be the existing e307 checkpoint")
    return validate_student(config["teacher"], config.get("student"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def row_key(row):
    payload = [row["input_token_ids"], row["output_token_ids"]]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


class Dataset:
    """Index JSONL offsets rather than retaining the entire trace corpus in RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open("rb") as f:
            while True:
                offset = f.tell()
                if not f.readline():
                    break
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("empty dataset")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as f:
            f.seek(self.offsets[index])
            row = json.loads(f.readline())
        if not row["input_token_ids"] or not row["output_token_ids"]:
            raise ValueError("empty prompt/completion")
        return row


def batch_indices(size, world, step, seed):
    if size < world:
        raise ValueError("dataset must contain at least one row per rank")
    batches = size // world
    epoch, batch = divmod(step, batches)
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch))
    return order[batch * world : (batch + 1) * world].tolist()


def validate_cache(cache, dataset, student_root, teacher_root=TEACHER):
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["dataset_sha256"] != sha(dataset):
        raise ValueError("teacher cache belongs to a different dataset")
    if manifest["teacher_identity"] != teacher_identity(teacher_root):
        raise ValueError("wrong teacher or teacher checkpoint changed")
    if manifest["tokenizer_sha256"] != sha(student_root / "tokenizer.json"):
        raise ValueError("teacher/student tokenizer mismatch")
    if sha(cache / "head.safetensors") != manifest["head_sha256"]:
        raise ValueError("teacher head changed")
    if manifest["rows"] != len(Dataset(dataset)):
        raise ValueError("incomplete teacher cache")
    return manifest
