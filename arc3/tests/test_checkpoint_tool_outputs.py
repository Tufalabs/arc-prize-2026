"""What the checkpoint tools under tools/checkpoints leave in a directory they
write: weights as readable as any other file, only the files serving reads
copied from the source, and a prune's record of what it was cut from.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from typing import Any

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools" / "checkpoints"
# What a conversion or an earlier prune leaves beside its weights.
SOURCE_LEFTOVERS = ("layer-0-experts-0.complete.json", "aime26_metrics.json", "README.md", "pruning.json")


def _load(name: str) -> Any:
    """A tool as a module: tools are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(root: pathlib.Path) -> pathlib.Path:
    """A one-shard checkpoint with an MTP block and a converted source's leftovers."""
    src = root / "source"
    src.mkdir()
    tensors = {
        "mtp.layers.0.mlp.gate.weight": torch.zeros(4, 3),
        "lm_head.weight": torch.zeros(5, 3),
        "model.language_model.embed_tokens.weight": torch.zeros(5, 3),
        "model.language_model.layers.0.weight": torch.zeros(2),
    }
    save_file(tensors, str(src / "model-00001-of-00001.safetensors"))
    index = {"weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")}
    (src / "model.safetensors.index.json").write_text(json.dumps(index))
    (src / "config.json").write_text(json.dumps({"text_config": {"num_experts": 4}}))
    for name in ("tokenizer.json", "chat_template.jinja", "hf_quant_config.json", "LICENSE", *SOURCE_LEFTOVERS):
        (src / name).write_text("{}")
    return src


@pytest.fixture
def group_umask() -> Any:
    """A shared tree's umask for the test, restored after it."""
    previous = os.umask(0o002)
    yield
    os.umask(previous)


def test_a_prunes_shard_has_the_mode_the_umask_gives(tmp_path: pathlib.Path, group_umask: None) -> None:
    prune = _load("prune_flashnext_experts")
    path = tmp_path / "shard.safetensors"
    prune.save_file({"a": torch.zeros(2)}, str(path), {"format": "pt"})
    assert path.stat().st_mode & 0o777 == 0o664
    assert os.umask(0o002) == 0o002


def test_a_prune_copies_what_serving_reads_and_names_its_source(tmp_path: pathlib.Path) -> None:
    prune = _load("prune_flashnext_experts")
    src = _source(tmp_path)
    out = tmp_path / "pruned"
    out.mkdir()
    assert prune.copy_serving_files(src, out) == [
        "chat_template.jinja",
        "tokenizer.json",
        "hf_quant_config.json",
        "LICENSE",
    ]
    assert sorted(p.name for p in out.iterdir()) == [
        "LICENSE",
        "chat_template.jinja",
        "hf_quant_config.json",
        "tokenizer.json",
    ]
    plan = {"num_experts": 4, "kept": {"0": [0, 1]}, "criterion": "reap"}
    record = prune.pruning_record(plan, src, "abc", 1)
    assert record == {**plan, "pruned_from": str(src.resolve()), "config_sha256": "abc", "shards": 1}


def test_an_extracted_head_holds_its_weights_and_what_serving_reads(
    tmp_path: pathlib.Path, group_umask: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    extract = _load("extract_mtp_drafter")
    src = _source(tmp_path)
    out = tmp_path / "head"
    monkeypatch.setattr(sys, "argv", ["extract_mtp_drafter.py", "--src", str(src), "--out", str(out)])
    extract.main()
    assert sorted(p.name for p in out.iterdir()) == [
        "chat_template.jinja",
        "config.json",
        "model.safetensors",
        "pruning.json",
        "tokenizer.json",
    ]
    assert (out / "model.safetensors").stat().st_mode & 0o777 == 0o664
    assert json.loads((out / "config.json").read_text())["text_config"]["mtp_num_experts"] == 4
